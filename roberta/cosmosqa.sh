
if [ "$1" = "train" ]; then
  export COSMOS_DIR=../../data/cosmosqa/
  python ./run_multiple_choice.py \
         --model_type roberta \
         --task_name cosmosqa \
         --model_name_or_path roberta-large \
         --do_train \
         --do_eval \
         --train_file train.jsonl \
         --eval_file dev.jsonl \
         --data_dir $SIQA_DIR \
         --learning_rate 5e-6 \
         --num_train_epochs 3 \
         --max_seq_length 128 \
         --output_dir ./baselines/siqa-roberta-large/ \
         --per_gpu_eval_batch_size=16 \
         --per_gpu_train_batch_size=2 \
         --gradient_accumulation_steps 4 \
         --eval_all_checkpoints \
         --overwrite_output \
         --save_steps 1000 \
        --logits_file logits.txt \
        --save_end_of_epoch

elif [ "$1" = "eval_valid" ]; then
  export COSMOSQA_DIR=../../data/cosmosqa/
  python ./run_multiple_choice.py \
         --model_type roberta \
         --task_name cosmosqa \
         --model_name_or_path ./baselines/cosmosqa-roberta-large/best/checkpoint-12000/ \
         --do_eval \
         --train_file train.jsonl \
         --eval_file train.jsonl \
         --data_dir $COSMOSQA_DIR \
         --learning_rate 5e-6 \
         --num_train_epochs 3 \
         --max_seq_length 128 \
         --output_dir ./baselines/cosmosqa-roberta-large/best/checkpoint-12000/ \
         --per_gpu_eval_batch_size=16 \
         --per_gpu_train_batch_size=2 \
         --gradient_accumulation_steps 4 \
         --save_steps 1000 \
        --logits_file train_logits.txt \
		--overwrite_cache

elif [ "$1" = "eval_test" ]; then
  export COSMOSQA_DIR=../../data/cosmosqa/
  python ./run_multiple_choice.py \
         --model_type roberta \
         --task_name cosmosqa \
         --model_name_or_path ./baselines/cosmosqa-roberta-large/bayes-5e-6-4-8/checkpoint-12000/ \
         --do_test \
         --train_file train.jsonl \
         --eval_file test.jsonl \
         --data_dir $COSMOSQA_DIR \
         --learning_rate 5e-6 \
         --num_train_epochs 3 \
         --max_seq_length 128 \
         --output_dir ./baselines/cosmosqa-roberta-large/bayes-5e-6-4-8/checkpoint-12000/ \
         --per_gpu_eval_batch_size=16 \
         --per_gpu_train_batch_size=2 \
         --gradient_accumulation_steps 4 \
         --eval_all_checkpoints \
         --save_steps 1000 \
        --logits_file logits.txt
fi