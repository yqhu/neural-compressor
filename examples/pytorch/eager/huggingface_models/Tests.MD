# Environment setup
<pre>
conda create -n lpot python=3.8
conda activate lpot
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install lpot

git clone https://github.com/yqhu/neural-compressor
cd neural-compressor/examples/pytorch/eager/huggingface_models
python setup.py install
</pre>

# Installations
<pre>
cd examples/text-classification
pip install -r requirements.txt

cd ../seq2seq
pip install -r requirements.txt
cd ../..
</pre>

# MRPC: Model preparation
<pre>
bash prepare_MRPC.sh
</pre>

# MRPC: LOP tuning
<pre>
bash tune_seq2seq.sh
</pre>

# MRPC: Evaluations
## Original model
<pre>
bash eval_tuning_original.sh
</pre>
## Native dynamic tuning by PyTorch
<pre>
bash eval_tuning_native.sh
</pre>


# seq2seq: Model preparation
<pre>
cd examples/seq2seq
bash prepare_MRPC.sh
cd ..
</pre>

# seq2seq: LOP tuning
<pre>
bash tune_seq2seq.sh
</pre>

# seq2seq: Evaluations
## Original model
<pre>
bash eval_tuning_seq2seq_original.sh
</pre>
## Native dynamic tuning by PyTorch
<pre>
bash eval_tuning_seq2seq_native.sh
</pre>
## LPOT tuned
<pre>
bash eval_tuning_seq2seq_lpot.sh
</pre>

# Accuracy vs Speed/Model size tradeoff
Modify conf.yaml to test different values for tuning->accuracy_criterion->relative.
