# Flow Matching Zero-Shot Classifier

## Requirements
You can install the environment by:
```
conda env create -f env.yaml
```

Also install our version of `diffusers`:
```
cd diffusers
pip install -e .
```

## Usage
1. Use `train.py` to train a model on MNIST now.
2. Use `eval.py` to view the generation examples.
3. Use `classifier.py` to test the classification performance.
4. Use `flux.py` to do zero-shot classification.
