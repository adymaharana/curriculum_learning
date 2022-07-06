#for TASK_NAME in cola sst2 mrpc stsb rte wnli qnli mnli qqp
for TASK_NAME in cola stsb rte
do
    for LR in 1e-5 2e-5 3e-5
	do
		for GRAD_ACC in 2 4
		do
			python run_glue.py \
				--model_name_or_path bert-large-uncased \
				--task_name $TASK_NAME \
				--max_length 128 \
				--per_device_train_batch_size 8 \
				--gradient_accumulation_steps $GRAD_ACC \
				--learning_rate $LR \
				--num_train_epochs 3 \
				--output_dir /nas-hdd/tarbucket/adyasha/models/glue_bert_large/${TASK_NAME}_${LR}_${GRAD_ACC}/ \
				--seed 1099
		done
	done
done
