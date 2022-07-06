# Model directory log
# total batch size = 16
if [ "$1" = "train" ]; then
	for i in 0 1 2 3 4
	do
	  export CODAH_DIR=../../data/codah/fold_$i/
	  python ./run_multiple_choice.py \
		--model_type roberta \
		--task_name codah \
		--model_name_or_path roberta-large \
		--do_train \
		--do_eval \
		--train_file train.csv \
		--eval_file dev.csv \
		--data_dir $CODAH_DIR \
		--learning_rate 1e-5 \
		--max_seq_length 90 \
		--output_dir ./baselines/codah-roberta-large/fold_$i/ \
		--per_gpu_eval_batch_size=16 \
		--per_gpu_train_batch_size=2 \
		--gradient_accumulation_steps 8 \
		--overwrite_output \
		--save_steps 100 \
		--warmup_steps 40 \
		--weight_decay 0.01 \
		--adam_epsilon 1e-6 \
		--num_train_epochs 5 \
		--logits_file logits.txt \
		--eval_all_checkpoints \
		--save_end_of_epoch
	done
elif [ "$1" = "eval_valid" ]; then
  for i in 0 1 2 3 4
	do
	  export CODAH_DIR=../../data/codah/fold_$i/
	  python ./run_multiple_choice.py \
		--model_type roberta \
		--task_name codah \
		--model_name_or_path ./baselines/codah-roberta-large/fold_$i/ \
		--do_eval \
		--train_file train.csv \
		--eval_file train.csv \
		--data_dir $CODAH_DIR \
		--learning_rate 1e-5 \
		--max_seq_length 90 \
		--output_dir ./baselines/codah-roberta-large/fold_$i/ \
		--per_gpu_eval_batch_size=16 \
		--per_gpu_train_batch_size=2 \
		--gradient_accumulation_steps 8 \
		--save_steps 100 \
		--warmup_steps 40 \
		--weight_decay 0.01 \
		--adam_epsilon 1e-6 \
		--num_train_epochs 5 \
		--logits_file train_logits.txt \
		--eval_all_checkpoints
	done
fi