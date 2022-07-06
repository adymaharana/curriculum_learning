from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
import logging, argparse
import numpy as np
import glob, os, shutil
import random, json

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMultipleChoice,
    BertTokenizer,
    RobertaConfig,
    RobertaForMultipleChoice,
    RobertaTokenizer,
    XLNetConfig,
    XLNetForMultipleChoice,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
#from transformers import WarmupLinearSchedule
from utils_multiple_choice import convert_examples_to_features, processors, get_qap
#from modelling_roberta_mcq import RobertaForMultipleChoice

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

# ALL_MODELS = sum(
#     (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, RobertaConfig)), ()
# )

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMultipleChoice, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer),
}


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        return data + (index, )

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

IndexedTensorDataset = dataset_with_indices(TensorDataset)

class SPLLoss(torch.nn.NLLLoss):
    def __init__(self, *args, device=torch.device("cpu"), n_samples=0, warmup_steps=500, **kwargs):
        super(SPLLoss, self).__init__(*args, **kwargs)
        self.threshold = 0.5
        self.growing_factor = 1.3
        self.v = torch.zeros(n_samples).int().to(device)
        self.warmup_steps = warmup_steps

    def forward(self, input: torch.Tensor, target: torch.Tensor, index: torch.Tensor, n_steps) -> torch.Tensor:

        super_loss = torch.nn.functional.nll_loss(torch.log_softmax(input, dim=-1), target, reduction="none")

        #if n_steps <= self.warmup_steps:
        #    return super_loss.mean()
        #else:
        v = self.spl_loss(super_loss)
        self.v[index] = v
        return (super_loss * v.float()).mean()

    def increase_threshold(self):
        self.threshold *= self.growing_factor

    def spl_loss(self, super_loss):
        v = super_loss < self.threshold
        return v.int()

    def save_weights(self):
        weights = self.v.detach().cpu().numpy()
        np.save('weights.npy', weights)


def train(args, train_dataset, model, tokenizer, train_examples=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    full_train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    full_train_dataloader = DataLoader(train_dataset, sampler=full_train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(full_train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(full_train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.self_paced:
        criterion = SPLLoss(device=args.device, n_samples=len(train_dataset))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_ratio*t_total), num_training_steps=t_total
    )
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    #global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_dev_loss = 1000.0
    best_steps = 0
    model.zero_grad()
    train_iterator = trange(int(t_total), desc="Steps", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility

    curr_idxs = list(range(0, int(args.starting_percent*len(train_dataset))))
    curr_dataloader = DataLoader(Subset(train_dataset, curr_idxs),
                                 sampler=RandomSampler(Subset(train_dataset, curr_idxs), replacement=False) if args.local_rank == -1 else DistributedSampler(Subset(train_dataset, curr_idxs)),
                                 batch_size=args.train_batch_size)
    curr_percent = args.starting_percent

    global_step = 0
    print('Starting with %s starting percent, %s increase factor and %s step length' % (args.starting_percent, args.increase_factor, args.step_length))
    while global_step <= t_total:

        print("Restarting dataloader")
        print("Length of current dataloader", len(curr_dataloader))
        data_iterator = tqdm(curr_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        print("Length of current iterator", len(data_iterator))
        
        for step, batch in enumerate(data_iterator):

            #print("In data iterator loop")
            
            sample_idxs = batch[-1]

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                # "token_type_ids": batch[2]
                # if args.model_type in ["bert", "xlnet"]
                # else None,  # XLM don't use segment_ids
                "labels": batch[2],
            }
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            # print(outputs[2].shape)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                #torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if results["eval_acc"] > best_dev_acc:
                        # if results["eval_loss"] <= best_dev_loss:
                            best_dev_acc = results["eval_acc"]
                            best_dev_loss = results["eval_loss"]
                            best_steps = global_step
                            if args.do_test:
                                results_test = evaluate(args, model, tokenizer, test=True)
                                for key, value in results_test.items():
                                    tb_writer.add_scalar("test_{}".format(key), value, global_step)
                                logger.info(
                                    "test acc: %s, loss: %s, global steps: %s",
                                    str(results_test["eval_acc"]),
                                    str(results_test["eval_loss"]),
                                    str(global_step),
                                )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info(
                        "Average loss: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str(global_step),
                    )
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Evaluate if checkpoint is better than previous checkpoint
                    results = evaluate(args, model, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    if results["eval_acc"] > best_dev_acc:
                    # if results["eval_loss"] <= best_dev_loss:
                        best_dev_acc = results["eval_acc"]
                        best_dev_loss = results["eval_loss"]
                        best_steps = global_step

                        # Delete existing checkpoints
                        checkpoint_dirs = glob.glob(os.path.join(args.output_dir, "checkpoint*"))
                        for dirname in checkpoint_dirs:
                            if os.path.isdir(dirname):
                                shutil.rmtree(dirname)
                        
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_vocabulary(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                if global_step % args.step_length == 0 and global_step > 0:

                    print(global_step)
                    data_iterator.close()
                    if curr_percent < 1:
                        if args.adaptive:
                            print("Recomputing rank using adaptive CL")
                            train_logits = evaluate(args, model, tokenizer, dataset= train_dataset, return_logits=True)
                            train_logits = np.array([np.array(logits) for logits in train_logits])
                            train_labels = [d[2].item() for d in train_dataset]
                            curr_qap = get_qap(train_logits, train_labels)
                            prev_qap = [d[3].item() for d in train_dataset]
                            new_qap = [(1 - args.alpha) * prev_qap[i] + (args.alpha * curr_qap[i]) for i in
                                       range(0, len(train_labels))]
                            # sort_index = torch.tensor(np.flip(np.argsort(new_qap)).tolist())
                            sort_index = torch.tensor(np.argsort(new_qap))
                            train_dataset = TensorDataset(
                                torch.index_select(train_dataset.tensors[0], dim=0, index=sort_index),
                                torch.index_select(train_dataset.tensors[1], dim=0, index=sort_index),
                                torch.index_select(train_dataset.tensors[2], dim=0, index=sort_index),
                                torch.tensor([new_qap[i.item()] for i in sort_index], dtype=torch.float))

                        curr_percent = min(args.starting_percent*(args.increase_factor**int(global_step/args.step_length)), 1)

                    curr_idxs = list(range(0, int(curr_percent * len(train_dataset))))

                    curr_dataloader = DataLoader(Subset(train_dataset, curr_idxs), sampler=RandomSampler(Subset(train_dataset, curr_idxs), replacement=False)
                                                 if args.local_rank == -1 else DistributedSampler(Subset(train_dataset, curr_idxs)), batch_size=args.train_batch_size)
                
                    print("At step %s, usage percent changed to %s" % (global_step, curr_percent))
                    print(len(curr_dataloader))
                    break


            if args.max_steps > 0 and global_step > args.max_steps:
                data_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    if args.self_paced:
        criterion.save_weights()

    return global_step, tr_loss / global_step, best_steps


def evaluate(args, model, tokenizer, prefix="", test=False, dataset=None, return_logits=False):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):

        if dataset is not None:
            eval_dataset = dataset
        else:
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=not test, test=test)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1 and not args.no_cuda and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        roberta_embeddings = []

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        all_logits = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    # "token_type_ids": batch[2]
                    # if args.model_type in ["bert", "xlnet"]
                    # else None,  # XLM don't use segment_ids
                    "labels": batch[2],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

                # print(batch[0].shape)
                # print(outputs[1].shape)
                # print(outputs[2].shape)
                # roberta_embeddings.append(outputs[2].cpu().numpy())

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            all_logits.extend(logits.detach().cpu().numpy().tolist())

        if return_logits:
            return all_logits

        # np.save('train_embeddings.npy', np.vstack(roberta_embeddings))

        #with open('logits.txt', 'w') as f:
        #    for logit in all_logits:
        #        f.write(json.dumps(logit) + '\n')

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        acc = simple_accuracy(preds, out_label_ids)
        result = {"eval_acc": acc, "eval_loss": eval_loss}
        results.update(result)

        with open('predictions.lst', 'w') as fresult:
            fresult.write('\n'.join([str(n + 1) for n in preds]))

        output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")

        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(str(prefix) + " is test:" + str(test)))
            writer.write("model           =%s\n" % str(args.model_name_or_path))
            writer.write(
                "total batch size=%d\n"
                % (
                        args.per_gpu_train_batch_size
                        * args.gradient_accumulation_steps
                        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
                )
            )
            writer.write("train num epochs=%d\n" % args.num_train_epochs)
            writer.write("fp16            =%s\n" % args.fp16)
            writer.write("max seq length  =%d\n" % args.max_seq_length)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write('\n\n')
    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False, return_examples=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    cached_features_file = os.path.join(
        args.output_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not return_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, args.eval_file)
        elif test:
            examples = processor.get_test_examples(args.data_dir, args.eval_file)
        else:
            examples = processor.get_train_examples(args.data_dir, args.train_file, args.train_logits_file)
            for example in random.sample(examples, k=5):
                print("QAP: ", example.qap, ", id: ", example.example_id, ", context: ", example.contexts,
                      ", question: ", example.question, ", options: ", example.endings)

            if args.curriculum_learning:
                examples = processor.sort_examples(examples, 'qap')
                logger.info("Sorted dataset for curriculum learning.")
                logger.info("*****  Sorted Examples *****")
                for example in random.sample(examples, k=5):
                    print("QAP: ", example.qap, ", id: ", example.example_id, ", context: ", example.contexts, ", question: ", example.question, ", options: ", example.endings)

        logger.info("Training number: %s", str(len(examples)))
        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            curriculum=args.curriculum_learning and not evaluate and not test
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    # all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    all_qap = torch.tensor([f.qap for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_qap)

    if return_examples:
        return dataset, examples
    else:
        return dataset

def roberta_train_and_eval(arg_string):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--train_file", default='', type=str, help="Training file.")
    parser.add_argument("--eval_file", default='', type=str, help="Evaluation file.")
    parser.add_argument("--train_logits_file", default='', type=str, help="Logits for Training file.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")

    parser.add_argument("--curriculum_learning", action="store_true", help="Whether to use curriculum learning.")
    parser.add_argument("--starting_percent", default=0.3, type=float, help="Starting percentage of training data for curriculum learning")
    parser.add_argument("--increase_factor", default=1.1, type=float, help="Multiplication factor for incrasing data usage after step length iterations")
    parser.add_argument("--step_length", default = 750, type=int, help="Number of iterations after which pacing function is updated")
    parser.add_argument("--adaptive", action="store_true", help="Whether to use adaptive curriculum learning.")
    parser.add_argument("--alpha", default=0.0, type=float, help="Update ratio for adaptive CL")

    parser.add_argument("--self_paced", action="store_true", help="Whether to use self-paced learning.")
    parser.add_argument("--threshold_update_steps", default=1000, type=int,
                        help="Update SPL self-pace parameter every x steps")

    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--log_file", type=str, default='log.jsonl', help="Log in this file in the output directory.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args(arg_string)

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Training
    if args.do_train:
        if args.adaptive:
            train_dataset, train_examples = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, return_examples=args.adaptive)
        else:
            train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
            train_examples = None
        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer, train_examples)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in
                sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        fresult = open(args.log_file, 'a+')
        best_eval = 0.0
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            if best_eval <= result['eval_acc']:
                best_eval = result['eval_acc']
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

            result['checkpoint'] = os.path.join(args.output_dir, checkpoint)
            result['starting_percent'] = args.starting_percent
            result['increase_factor'] = args.increase_factor
            result['step_length'] = args.step_length
            result['learning_rate'] = args.learning_rate
            result['alpha'] = args.alpha

            fresult.write(json.dumps(result) + '\n')
        fresult.close()

    #return -results['eval_loss_']
    #return results['eval_acc_']
    return best_eval

def optimize_roberta(args):
    """Apply Bayesian Optimization to SVC parameters."""

    if args.cl:
        def roberta_wrapper(starting_percent, inc, step_length):
            """Wrapper of RoBERTa training and validation.

            Notice how we ensure all parameters are casted
            to integer before we pass them along. Moreover, to avoid max_features
            taking values outside the (0, 1) range, we also ensure it is capped
            accordingly.
            """

            starting_percent = round(starting_percent, 2)
            inc = round(inc, 2)
            step_length = int(250*(step_length))
            base_args_list = ['--model_type', 'roberta',
                              '--task_name', 'winogrande',
                              '--model_name_or_path', 'roberta-large',
                              '--train_file', 'train_%s.jsonl' % args.size,
                              '--eval_file', 'dev.jsonl',
                              '--train_logits_file', './baselines/winogrande-%s-roberta-large/_train_logits.txt' % args.size,
                              '--data_dir', '../data/winogrande/',
                              '--output_dir', './out/winogrande-' + args.size + '-roberta-large/qap-cl-' + str(starting_percent) + '-' + str(inc) + '-' + str(step_length),
                              '--logging_steps', '100',
                              '--log_file', 'winogrande_%s_qap_cl.jsonl' % args.size,
                              '--do_train', '--do_eval', '--curriculum_learning',
                              '--learning_rate', '1e-5',
                              '--num_train_epochs', '5',
                              '--max_seq_length', '70',
                              '--save_steps', '1000',
                              '--overwrite_output',
                              '--per_gpu_eval_batch_size', '8',
                              '--per_gpu_train_batch_size', '4',
                              '--gradient_accumulation_steps', '4',
                              '--overwrite_cache']

            opt_args_list = ['--starting_percent', str(starting_percent),
                             '--increase_factor', str(inc),
                             '--step_length', str(step_length)]

            return roberta_train_and_eval(base_args_list + opt_args_list)

        # For curriculum learning experiments CQA, CNET merged. Step length multiplier = 500*slen
        optimizer = BayesianOptimization(
            f=roberta_wrapper,
            pbounds={"starting_percent": (0.01, 0.5),
                     "inc": (1.05, 2.0),
                     "step_length": (0.01, 6)},
            random_state=1234,
            verbose=2
        )

        optimizer.maximize(
            init_points=3,
            n_iter=15
            # What follows are GP regressor parameters
            # alpha=1e-3,
            # n_restarts_optimizer=5
        )
        print("Final result:", optimizer.max)


    elif args.acl:
        def roberta_wrapper(starting_percent, inc, step_length, alpha):
            """Wrapper of RoBERTa training and validation.

            Notice how we ensure all parameters are casted
            to integer before we pass them along. Moreover, to avoid max_features
            taking values outside the (0, 1) range, we also ensure it is capped
            accordingly.
            """

            starting_percent = round(starting_percent, 2)
            inc = round(inc, 2)
            if args.size in ['xl']:
                step_length = 1000 + int(250*(step_length))
            else:
                step_length = int(250 * (step_length))

            alpha = round(alpha, 2)

            base_args_list = ['--model_type', 'roberta',
                              '--task_name', 'winogrande',
                              '--model_name_or_path', 'roberta-large',
                              '--train_file', 'train_%s.jsonl' % args.size,
                              '--eval_file', 'dev.jsonl',
                              '--train_logits_file', './baselines/winogrande-%s-roberta-large/_train_logits.txt' % args.size,
                              '--data_dir', '../data/winogrande/',
                              '--output_dir', './out/winogrande-' + args.size + '-roberta-large/qap-cl-' + str(starting_percent) + '-' + str(inc) + '-' + str(step_length) + '-' + str(alpha),
                              '--logging_steps', '100',
                              '--log_file', 'winogrande_%s_qap_cl.jsonl' % args.size,
                              '--do_train', '--do_eval', '--curriculum_learning', '--adaptive',
                              '--learning_rate', '1e-5',
                              '--num_train_epochs', '5',
                              '--max_seq_length', '70',
                              '--save_steps', '1000',
                              '--overwrite_output',
                              '--per_gpu_eval_batch_size', '8',
                              '--per_gpu_train_batch_size', '4',
                              '--gradient_accumulation_steps', '4']

            opt_args_list = ['--starting_percent', str(starting_percent),
                             '--increase_factor', str(inc),
                             '--step_length', str(step_length),
                             '--alpha', str(alpha)]

            return roberta_train_and_eval(base_args_list + opt_args_list)

        # For curriculum learning experiments CQA, CNET merged. Step length multiplier = 500*slen
        optimizer = BayesianOptimization(
            f=roberta_wrapper,
            pbounds={"starting_percent": (0.01, 0.5),
                     "inc": (1.05, 2.0),
                     "step_length": (0.01, 6),
                     "alpha": (0, 1)},
            random_state=1234,
            verbose=2
        )

        optimizer.maximize(
            init_points=3,
            n_iter=15
            # What follows are GP regressor parameters
            # alpha=1e-3,
            # n_restarts_optimizer=5
        )
        print("Final result:", optimizer.max)

    else:
        raise ValueError

if __name__ == "__main__":

    main_parser = argparse.ArgumentParser()

    # Required parameters
    main_parser.add_argument("--cl", action="store_true", help="Whether to optimize pacing function")
    main_parser.add_argument("--acl", action="store_true", help="Whether to optimize pacing function with adaptive CL.")
    main_parser.add_argument("--hp", action="store_true", help="Whether to optimize hyperparams")
    main_parser.add_argument("--size", type=str, help="Choose Winogrande data size from xl, l, m, s, xs")
    main_args = main_parser.parse_args()

    assert main_args.cl or main_args.acl or main_args.hp, "Select only one setting for optimization"

    print(Colours.yellow("--- Optimizing Roberta ---"))
    optimize_roberta(main_args)