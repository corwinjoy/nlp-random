# nlp-random
ModernBERT deterministic training investigation

# Installation
You will need to install the latest version of the packages `datasets` and `transformers` from source which are listed as submodules.

```
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
cd transformers
pip install .[tf,torch,sentencepiece,vision,optuna,sklearn,onnxruntime]
cd ..
cd datasets
pip install -e ".[quality]"
cd ..
pip install flash-attn --no-build-isolation
pip install -f requirements.txt
```

# Usage
Run `deterministic_training.py` after activating the virtual environment.

