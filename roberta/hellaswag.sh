# Model directory log

# batch size = 8
if [ "$1" = "train" ]; then
  export DATA_DIR=../../data/hellaswag
  python ./run_multiple_choice.py \
         --model_type roberta \
         --task_name hellaswag \
         --model_name_or_path roberta-large \
         --do_train \
         --do_eval \
         --train_file hellaswag_2k_train.jsonl \
         --eval_file hellaswag_2k_val.jsonl \
         --data_dir $DATA_DIR \
         --learning_rate 1e-5 \
         --max_seq_length 128 \
         --output_dir ./baselines/hellaswag-2k-roberta-large/baseline/ \
         --per_gpu_eval_batch_size=16 \
         --per_gpu_train_batch_size=2 \
         --gradient_accumulation_steps 4 \
         --overwrite_output \
         --logging_steps 50 \
         --save_steps 100 \
         --warmup_steps 15 \
         --weight_decay 0.01 \
         --num_train_epochs 5 \
         --save_end_of_epoch \
         --eval_all_checkpoints

elif [ "$1" = "eval_valid" ]; then
  export DATA_DIR=../../data/hellaswag
  python ./run_multiple_choice.py \
         --model_type roberta \
         --task_name hellaswag \
         --model_name_or_path /nas-hdd/tarbucket/adyasha/models/hellaswag-roberta-large/qap-cl-0.36-1.71-187-0.07/ \
         --do_eval \
         --train_file hellaswag_2k_train.jsonl \
         --eval_file hellaswag_2k_train.jsonl \
         --data_dir $DATA_DIR \
         --learning_rate 1e-5 \
         --max_seq_length 128 \
         --output_dir /nas-hdd/tarbucket/adyasha/models/hellaswag-roberta-large/qap-cl-0.36-1.71-187-0.07/ \
         --per_gpu_eval_batch_size=16 \
         --per_gpu_train_batch_size=2 \
         --gradient_accumulation_steps 8 \
		 --logits_file train_logits.txt \
		 --overwrite_cache
fi