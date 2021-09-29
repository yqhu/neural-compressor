wget https://cdn-datasets.huggingface.co/summarization/pegasus_data/billsum.tar.gz
tar -xzvf billsum.tar.gz
cd billsum
ln -s validation.source val.source
ln -s validation.target val.target
cd ..

export TASK_NAME=summarization_billsum

python run_seq2seq_tune.py \
  --model_name_or_path google/pegasus-billsum \
  --do_train \
  --do_eval \
  --task $TASK_NAME \
  --data_dir ./billsum \
  --output_dir ./billsum_tuned \
  --overwrite_output_dir \
  --predict_with_generate \
  --max_source_length 1024 \
  --max_target_length=256 \
  --val_max_target_length=256 \
  --test_max_target_length=256

