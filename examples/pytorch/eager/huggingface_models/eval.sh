tuned_checkpoint=$1
model_name_or_path=./MRPC
TASK_NAME='MRPC'
MAX_SEQ_LENGTH=128
batch_size=16
input_model=./MRPC

python -u examples/text-classification/run_glue_tune.py \
    --tuned_checkpoint ${tuned_checkpoint} \
    --model_name_or_path ${model_name_or_path} \
    --task_name ${TASK_NAME} \
    --do_eval --benchmark --int8  \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_gpu_eval_batch_size ${batch_size} \
    --no_cuda \
    --output_dir ${input_model}

ls -l $1/best_model_weights.pt
