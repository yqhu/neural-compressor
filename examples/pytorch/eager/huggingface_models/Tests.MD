# Environment setup
<pre>
conda create -n lpot python=3.8
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install lpot

git clone https://github.com/yqhu/neural-compressor
cd neural-compressor/examples/pytorch/eager/huggingface_models
python setup.py install

cd examples/text-classification
pip install -r requirements.txt
</pre>

# Installations
<pre>
cd examples/text-classification
pip install -r requirements.txt

cd ../seq2seq
pip install -r requirements.txt
cd ../..
</pre>

# Model preparation
<pre>
bash prepare_MRPC.sh
</pre>

# LOP tuning
<pre>
bash tune_MRPC.sh
</pre>

# Evaluations
## Original model
<pre>
bash eval_tuning_original.sh
</pre>
## Native dynamic tuning by PyTorch
<pre>
bash eval_tuning_native.sh
</pre>
## LPOT tuned
<pre>
bash eval_tuning_lpot.sh
</pre>

