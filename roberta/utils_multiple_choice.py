# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

import csv
import glob
import json
import logging
import os
from typing import List
import random
import tqdm
import numpy as np
import xml.etree.ElementTree as etree

from transformers import PreTrainedTokenizer
logger = logging.getLogger(__name__)

def parse_logits_file(logits_file):

    with open(logits_file, 'r') as f:
        logits = [json.loads(line.strip()) for line in f.readlines()]
    return np.array([np.array(l) for l in logits])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.tile(np.sum(np.exp(x), axis=1), (x.shape[1], 1)).transpose()

def get_qap(logits, labels):

    probs = softmax(logits)
    probs_qap = []
    for i in range(logits.shape[0]):
        label_idx = labels[i]
        probs_qap.append(probs[i, label_idx])
    probs_qap = np.array(probs_qap)
    return probs_qap

class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None, qap=None, energy=None, variability=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label
        self.qap = qap
        self.energy = energy
        self.variability = variability

class InputFeatures(object):
    def __init__(self, example_id, choices_features, label, qap=0.0):
        self.example_id = example_id
        self.choices_features = [
            # {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            # for input_ids, input_mask, segment_ids in choices_features
            {"input_ids": input_ids, "input_mask": input_mask}
            for input_ids, input_mask in choices_features
        ]
        self.label = label
        self.qap = qap

        
class T5InputFeatures(object):
    def __init__(self, example_id, input, label):
        self.example_id = example_id
        self.features = {"source_ids": input["input_ids"].squeeze(),
                         "source_mask": input["attention_mask"].squeeze(),
                         "target_ids": label["input_ids"].squeeze(),
                         "target_mask": label["attention_mask"].squeeze()
        }
        self.label = label



class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def sort_examples(self, examples, key):
        """Sorts examples by the given key"""
        if key == 'qap':
            return sorted(examples, key=lambda x: x.qap, reverse=True)
        elif key == 'energy':
            return sorted(examples, key=lambda x: x.energy, reverse=True)
        elif key == 'variability':
            return sorted(examples, key=lambda x: x.variability)
        else:
            raise ValueError

    def update_examples(self, examples, logits, alpha):
        """Sorts examples by the given key"""

        assert len(examples) == len(logits)
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        label_idxs = [label_map[example.label] for example in examples]
        curr_qap = get_qap(logits, label_idxs)

        updated_examples = []
        for i, example in enumerate(examples):
            example.qap = (1-alpha)*example.qap + (alpha*curr_qap[i])
            updated_examples.append(example)

        return updated_examples


class HellaSwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def __init__(self):
        super(HellaSwagProcessor, self).__init__()

    def get_train_examples(self, data_dir, data_file, logits_file=None):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)), logits_file)

    def get_dev_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)))

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines: List[List[str]], logits_file=None):
        """Creates examples for the training and dev sets."""

        data = [json.loads(line.strip("\n")) for line in lines]

        if logits_file:
            print("------------------ Parsing logits file ----------")
            logits_best = parse_logits_file(logits_file)
            label_idxs = [d['label'] for d in data]
            qap = get_qap(logits_best, label_idxs)
        else:
            qap = None

        examples = []
        for i, data_raw in tqdm.tqdm(enumerate(data), desc="read hellaswag data"):
            # data_raw = json.loads(line.strip("\n"))

            label = str(data_raw["label"])

            if qap is not None:
                examples.append(InputExample(
                    example_id=data_raw["ind"],
                    question=data_raw["ctx"],
                    contexts=['', '', '', ''],
                    endings=data_raw["endings"],
                    label=label,
                    qap=qap[i],
                    )
                )
            else:
                examples.append(InputExample(
                    example_id=data_raw["ind"],
                    question=data_raw["ctx"],
                    contexts=['', '', '', ''],
                    endings=data_raw["endings"],
                    label=label,)
                )

        return examples


class CodahProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, data_file, logits_file=None):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, data_file)), "train", logits_file)

    def get_dev_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, data_file)), "dev")

    def get_test_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, data_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = list(csv.reader(f, delimiter=','))
        if lines[0][1] == 'id':
            return lines[1:]
        else:
            return lines

    def _create_examples(self, lines: List[List[str]], type: str, logits_file=None):
        """Creates examples for the training and dev sets."""

        if logits_file:
            logits_best = parse_logits_file(logits_file)
            label_idxs = [0] * len(lines)
            qap = get_qap(logits_best, label_idxs)
        else:
            qap = None

        examples = []
        for i, line in enumerate(lines):
            correct_answer = line[3]
            options = [line[3], line[4], line[5], line[6]]
            if type == 'train':
                random.shuffle(options)
                label = [k for k, opt in enumerate(options) if opt == correct_answer][0]
            else:
                label = 0
            if type == "train" and qap is not None:
                examples.append(InputExample(
                    example_id=i,
                    question=line[1],  # in the swag dataset, the
                    # common beginning of each
                    # choice is stored in "sent2".
                    contexts=['', '', '', ''],
                    endings=options,
                    label=str(label),
                    qap=qap[i],
                    )
                )
            else:
                examples.append(InputExample(
                    example_id=i,
                    question=line[1],  # in the swag dataset, the
                    # common beginning of each
                    # choice is stored in "sent2".
                    contexts=['', '', '', ''],
                    endings=options,
                    label=str(label),
                    )
                )

        return examples


class SiqaProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir, data_file, logits_file=None):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)),
                                     self._read_txt(os.path.join(data_dir, data_file.replace('.jsonl', '-labels.lst'))),
                                     "train", logits_file)
        #return self._create_examples(self._read_json(os.path.join(data_dir, "train_merged-ranked-by-qap.jsonl")),
        #                             self._read_txt(os.path.join(data_dir, "train_merged-ranked-by-qap-labels.lst")),
        #                             "train")

    def get_dev_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)),
                                     self._read_txt(os.path.join(data_dir, data_file.replace('.jsonl', '-labels.lst'))),
                                     "dev")
        #return self._create_examples(self._read_json(os.path.join(data_dir, "dev_merged.jsonl")),
        #                             self._read_txt(os.path.join(data_dir, "dev-labels.lst")),
        #                             "dev")

    def get_test_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)), None, "test")
        #return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")),
        #                             self._read_txt(os.path.join(data_dir, "test-labels.lst")),
        #                             "test")

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _read_txt(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, labels=None, type='train', logits_file=None):
        """Creates examples for the training and dev sets."""

        # check that lines and labels are of equal length
        if labels:
            assert len(lines) == len(labels), "%s samples and %s corresponding labels found in dataset" % (len(lines), len(labels))
        else:
            labels = ["1"]*len(lines)

        if logits_file:
            logits_best = parse_logits_file(logits_file)
            label_idxs = [int(label) - 1 for label in labels]
            qap = get_qap(logits_best, label_idxs)
        else:
            qap = None

        examples = []
        counter = 0
        for line, label in tqdm.tqdm(zip(lines, labels), desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            question = data_raw["question"]
            id = 'siqa-' + str(counter)
            options = [data_raw["answerA"], data_raw["answerB"], data_raw["answerC"]]
            context = data_raw["context"]
            truth = label.strip()
            if type == "train" and qap is not None:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[context for _ in options],
                        endings=[options[0], options[1], options[2]],
                        label=truth,
                        qap=qap[counter]
                    )
                )
            else:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[context for _ in options],
                        endings=[options[0], options[1], options[2]],
                        label=truth
                    )
                )
            counter += 1

        return examples



class ANLIProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir, data_file, logits_file=None):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)),
                                     self._read_txt(os.path.join(data_dir, data_file.replace('.jsonl', '-labels.lst'))),
                                     "train", logits_file)
        #return self._create_examples(self._read_json(os.path.join(data_dir, "train_merged-ranked-by-qap.jsonl")),
        #                             self._read_txt(os.path.join(data_dir, "train_merged-ranked-by-qap-labels.lst")),
        #                             "train")

    def get_dev_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)),
                                     self._read_txt(os.path.join(data_dir, data_file.replace('.jsonl', '-labels.lst'))),
                                     "dev")
        #return self._create_examples(self._read_json(os.path.join(data_dir, "dev_merged.jsonl")),
        #                             self._read_txt(os.path.join(data_dir, "dev-labels.lst")),
        #                             "dev")

    def get_test_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)), None, "test")
        #return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")),
        #                             self._read_txt(os.path.join(data_dir, "test-labels.lst")),
        #                             "test")

    def get_labels(self):
        """See base class."""
        return ["1", "2"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _read_txt(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, labels=None, type='train', logits_file=None):
        """Creates examples for the training and dev sets."""

        # check that lines and labels are of equal length
        if labels:
            assert len(lines) == len(labels), "%s samples and %s corresponding labels found in dataset" % (len(lines), len(labels))
        else:
            labels = ["1"]*len(lines)

        if logits_file:
            logits_best = parse_logits_file(logits_file)
            label_idxs = [int(label) - 1 for label in labels]
            qap = get_qap(logits_best, label_idxs)
        else:
            qap = None

        examples = []
        counter = 0
        for line, label in tqdm.tqdm(zip(lines, labels), desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            obs1 = data_raw["obs1"]
            obs2 = data_raw["obs2"]
            hyp1 = data_raw["hyp1"]
            hyp2 = data_raw["hyp2"]
            id = 'anli-' + str(counter)
            options = [hyp1, hyp2]
            truth = label.strip()
            if type == "train" and qap is not None:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=obs1,
                        contexts=['' for _ in options],
                        endings=[options[0] + ' ' + obs2, options[1] + ' ' + obs2],
                        label=truth,
                        qap=qap[counter],
                    )
                )
            else:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=obs1,
                        contexts=['' for _ in options],
                        endings=[options[0] + ' ' + obs2, options[1] + ' ' + obs2],
                        label=truth,
                    )
                )
            counter += 1
        return examples


class CQAProcessor(DataProcessor):
    """Processor for the CommonsenseQA data set."""

    def get_train_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)), "train")

    def get_dev_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)), "dev")

    def get_test_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["A", "B", "C", "D","E"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, mode):
        """Creates examples for the training and dev sets."""

        examples = []
        for line in tqdm.tqdm(lines, desc="read cqa data"):
            data_raw = json.loads(line.strip("\n"))
            question = data_raw["question"]["stem"]
            id = data_raw["id"]
            options = sorted(data_raw["question"]["choices"], key=lambda d: d["label"])
            options = [o["text"] for o in options]
            context = ''
            truth = data_raw["answerKey"]
            if mode == "train" and "qap" in data_raw:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[context for _ in options],
                        endings=[options[0], options[1], options[2], options[3], options[4]],
                        label=truth,
                        qap=data_raw["qap"],
                        energy=data_raw["energy"],
                        variability=data_raw["variability"]
                    )
                )
            else:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[context for _ in options],
                        endings=[options[0], options[1], options[2], options[3], options[4]],
                        label=truth,
                    )
                )
        return examples


class CosmosQAProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir, data_file, logits_file=None):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)),
                                     "train", logits_file)

    def get_dev_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)), "dev")

    def get_test_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type, logits_file=None):
        """Creates examples for the training and dev sets."""

        examples = []
        data = [json.loads(line.strip("\n")) for line in lines]

        if logits_file:
            print("------------------Found logits file-------------------")
            logits_best = parse_logits_file(logits_file)
            label_idxs = [int(d['label']) for d in data]
            qap = get_qap(logits_best, label_idxs)
        else:
            qap = None

        counter = 0
        for i, data_raw in tqdm.tqdm(enumerate(data), desc="read cosmosqa data"):

            question = data_raw["question"]
            id = data_raw["id"]
            options = [data_raw["answer0"], data_raw["answer1"], data_raw["answer2"], data_raw["answer3"]]
            context = data_raw["context"]
            if "label" in data_raw:
                truth = data_raw["label"]
            else:
                truth = "0"
            if type == "train" and qap is not None:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[context for _ in options],
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                        qap=qap[counter],
                    )
                )
            else:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[context for _ in options],
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
            counter += 1
        return examples


class WinograndeProcessor(DataProcessor):
    """Processor for the WinoGrande data set."""

    def get_train_examples(self, data_dir, data_file, logits_file=None):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)),
                                     self._read_txt(os.path.join(data_dir, data_file.replace('.jsonl', '-labels.lst'))),
                                     "train", logits_file)
        #return self._create_examples(self._read_json(os.path.join(data_dir, "train_merged-ranked-by-qap.jsonl")),
        #                             self._read_txt(os.path.join(data_dir, "train_merged-ranked-by-qap-labels.lst")),
        #                             "train")

    def get_dev_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)),
                                     self._read_txt(os.path.join(data_dir, data_file.replace('.jsonl', '-labels.lst'))),
                                     "dev")
        #return self._create_examples(self._read_json(os.path.join(data_dir, "dev_merged.jsonl")),
        #                             self._read_txt(os.path.join(data_dir, "dev-labels.lst")),
        #                             "dev")

    def get_test_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_json(os.path.join(data_dir, data_file)), None, "test")
        #return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")),
        #                             self._read_txt(os.path.join(data_dir, "test-labels.lst")),
        #                             "test")

    def get_labels(self):
        """See base class."""
        return ["1", "2"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _read_txt(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, labels=None, type='train', logits_file=None):
        """Creates examples for the training and dev sets."""

        # check that lines and labels are of equal length
        if labels:
            assert len(lines) == len(labels), "%s samples and %s corresponding labels found in dataset" % (len(lines), len(labels))
        else:
            labels = ["1"]*len(lines)

        if logits_file:
            logits_best = parse_logits_file(logits_file)
            label_idxs = [int(label) - 1 for label in labels]
            qap = get_qap(logits_best, label_idxs)
        else:
            qap = None

        examples = []
        counter = 0
        for line, label in tqdm.tqdm(zip(lines, labels), desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            question = data_raw["sentence"]
            id = 'winogrande-' + str(counter)
            options = [data_raw["option1"], data_raw["option2"]]
            context = ''
            truth = label.strip()
            if type == "train" and qap is not None:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[context for _ in options],
                        endings=[options[0], options[1]],
                        label=truth,
                        qap=qap[counter]
                    )
                )
            else:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[context for _ in options],
                        endings=[options[0], options[1]],
                        label=truth
                    )
                )
            counter += 1

        return examples


class WinoGradProcessor(DataProcessor):
    """Processor for the WinoGrande data set."""

    def get_dev_examples(self, data_dir, data_file):
        """See base class."""
        logger.info("LOOKING AT %s directory and %s file" % (data_dir, data_file))
        return self._create_examples(self._read_xml(os.path.join(data_dir, data_file)), "dev")

    def get_labels(self):
        """See base class."""
        return ["A", "B"]

    def _read_xml(self, input_file):
        # import data from WSCollection.xml
        tree = etree.parse(input_file)
        root = tree.getroot()
        problems = list()
        original_problems = root.getchildren()
        # parse original_problems to problem and append it to problems list
        for original_problem in original_problems:
            problem = dict()
            for information in original_problem.getchildren():
                if information.tag == 'answers':
                    answers = information.getchildren()
                    answer_list = list()
                    for answer in answers:
                        answer_list.append(answer.text.strip())
                    problem['answers'] = answer_list
                elif information.tag == 'text':
                    texts = information.getchildren()
                    text_dict = dict()
                    for text1 in texts:
                        text_dict[text1.tag] = text1.text.replace('\n', ' ').strip()
                        # text_dict[text1.tag] = text1.text
                    problem['text'] = text_dict
                elif information.tag == 'quote':
                    pass
                else:
                    problem[information.tag] = information.text.replace(' ', '')
            problems.append(problem)
        return problems

    def _create_examples(self, lines, type='train'):
        """Creates examples for the training and dev sets."""

        examples = []
        counter = 0
        for line in tqdm.tqdm(lines, desc="read arc data"):
            question = line["text"]["txt1"] + ' _ ' + line["text"]["txt2"]
            id = 'winograd-' + str(counter)
            options = line["answers"]
            context = ''
            truth = line["correctAnswer"].replace('.', '')
            examples.append(
                InputExample(
                    example_id=id,
                    question=question,
                    contexts=[context for _ in options],
                    endings=[options[0], options[1]],
                    label=truth
                )
            )
            counter += 1

        return examples


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")

        examples = [
            InputExample(
                example_id=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11],
            )
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples


class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[
                            options[0]["para"].replace("_", ""),
                            options[1]["para"].replace("_", ""),
                            options[2]["para"].replace("_", ""),
                            options[3]["para"].replace("_", ""),
                        ],
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth,
                    )
                )

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
    curriculum=False
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length, truncation=True)
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            # input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            input_ids = inputs["input_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                # token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                # token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            # assert len(token_type_ids) == max_length
            # choices_features.append((input_ids, attention_mask, token_type_ids))
            choices_features.append((input_ids, attention_mask))

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (input_ids, attention_mask) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
                # logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
                logger.info("label: {}".format(label))

        if curriculum:
            features.append(
                InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label, qap=example.qap))
        else:
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label,))

    print("curriculum is", curriculum)

    return features


def convert_examples_to_features_for_t5(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        
        input_ = example.contexts[0]
        options = ['%s: %s' % (i, option) for i, option in zip(label_list, example.endings)]
        options = " ".join(options)
        
        if example.question != '':
            input_ = "cqa context: %s  question: %s options: %s </s>" % (input_, example.question, options)
        else:
            input_ = "cqa context: %s options: %s </s>" % (input_, options)
            
        target = "%s </s>" % example.label

        # tokenize inputs
        tokenized_inputs = tokenizer.batch_encode_plus(
            [input_], max_length=max_length, pad_to_max_length=True, return_tensors="pt"
        )
        # tokenize targets
        tokenized_targets = tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt"
        )
        
        features.append(T5InputFeatures(example_id=example.example_id, input=tokenized_inputs, label=tokenized_targets,))

    return features


processors = {"race": RaceProcessor,
              "swag": SwagProcessor,
              "arc": ArcProcessor,
              "siqa": SiqaProcessor,
              "cqa": CQAProcessor,
              "codah": CodahProcessor,
              "hellaswag": HellaSwagProcessor,
              "cosmosqa": CosmosQAProcessor,
              "anli": ANLIProcessor,
              "winogrande": WinograndeProcessor,
              "wsc": WinoGradProcessor}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race", 4,
                                    "swag", 4,
                                    "arc", 4,
                                    "siqa", 3,
                                    "cqa", 5,
                                    "codah", 4,
                                    "hellaswag", 4,
                                    "cosmosqa", 4,
                                    "anli", 2,
                                    "winogrande", 2,
                                    "wsc", 2}
