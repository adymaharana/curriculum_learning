if [ "$1" = "train" ]; then
  export SIQA_DIR=../../data/siqa/
  python ./run_multiple_choice.py \
         --model_type roberta \
         --task_name siqa \
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
  export SIQA_DIR=../../data/siqa/
  python ./run_multiple_choice.py \
         --model_type roberta \
         --task_name siqa \
         --model_name_or_path ./out/siqa-roberta-large/qap-cl-0.5-1.92-534/ \
         --do_eval \
         --train_file train.jsonl \
         --eval_file dev.jsonl \
         --data_dir $SIQA_DIR \
         --learning_rate 5e-6 \
         --num_train_epochs 3 \
         --max_seq_length 128 \
         --output_dir ./out/siqa-roberta-large/qap-cl-0.5-1.92-534/ \
         --per_gpu_eval_batch_size=16 \
         --per_gpu_train_batch_size=2 \
         --gradient_accumulation_steps 4 \
         --save_steps 1000 \
        --logits_file val_logits.txt \
		--overwrite_cache

elif [ "$1" = "eval_test" ]; then
  export SIQA_DIR=../../data/siqa/
  python ./run_multiple_choice.py \
         --model_type roberta \
         --task_name siqa \
         --model_name_or_path ./out/siqa-roberta-large/qap-cl-0.5-1.92-534/ \
         --do_test \
         --train_file train.jsonl \
         --eval_file socialiqa.jsonl \
         --data_dir $SIQA_DIR \
         --learning_rate 5e-6 \
         --num_train_epochs 3 \
         --max_seq_length 128 \
         --output_dir ./out/siqa-roberta-large/qap-cl-0.5-1.92-534/ \
         --per_gpu_eval_batch_size=16 \
         --per_gpu_train_batch_size=2 \
         --gradient_accumulation_steps 4 \
         --eval_all_checkpoints \
         --save_steps 1000 \
        --logits_file logits.txt
fi
