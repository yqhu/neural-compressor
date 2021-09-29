tuned_checkpoint=saved_results_billsum_intel
model_name_or_path=./examples/seq2seq/billsum_tuned
TASK_NAME='billsum'
batch_size=16
input_model=./examples/seq2seq/billsum_tuned

python -u examples/seq2seq/run_seq2seq_tune.py \
    --tuned_checkpoint ${tuned_checkpoint} \
    --model_name_or_path ${model_name_or_path} \
    --data_dir ./examples/seq2seq/billsum \
    --task ${TASK_NAME} \
    --do_eval --benchmark --int8 \
    --predict_with_generate \
    --per_gpu_eval_batch_size ${batch_size} \
    --no_cuda \
    --output_dir ${input_model}

