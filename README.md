# Flow Matching Zero-Shot Classifier

## TODO
1. See the https://github.com/atong01/conditional-flow-matching/tree/main/examples/images to train cifar10 flow matching with **class conditon**.
**Note:** To make it class conditioned, you need to modify the code, set `class_cond=True` and `num_classes=10` for UnetWrapper, also modifiy the dataloader.
2. Get FLUX work


## Requirements
Only `torchcfm` needed now.

## Usage
1. Use `train.py` to train a model on MNIST now.
2. Use `eval.py` to view the generation examples.
3. Use `classifier.py` to test the classification performance.
