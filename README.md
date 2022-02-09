#  📝 ⇝ 🧏
## Sign Translate
Use sequnce to sequence deep neural nets to translate many languages to sign language


### setup environment:
```bash
conda create --name hf --no-default-packages python=3.8
conda activate hf
# install torch
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# install huggingface
pip install datasets transformers tokenizers pyarrow==3.0.0 
# install others
pip install sentencepiece sacrebleu 
```

### how to run:
1. run `bash scripts/prepare_model.sh` to prepare model
1. run `bash scripts/prepare_data.sh` to prepare data
1. run `bash scripts/train.sh` to train model