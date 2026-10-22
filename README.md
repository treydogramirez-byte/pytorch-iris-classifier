# Basic PyTorch Iris Classifier

A small neural network built with PyTorch that predicts iris flower species from four measurements. This follows the instructor live build guide, from a tensor and gradient demo through a trained and evaluated classifier.

## What the notebook does

- Loads the 150-row iris dataset and splits it into 90 training, 30 validation, and 30 untouched test rows
- Shows how a tensor stores numbers and how autograd computes a gradient
- Reads a `Linear(4, 8)` layer: 32 weights plus 8 biases, 40 learned parameters
- Compares ReLU, Leaky ReLU, sigmoid, and tanh on the same three inputs
- Trains a `Linear(4, 8)`, ReLU, `Linear(8, 3)` network with CrossEntropyLoss and Adam
- Compares hidden width 8 against hidden width 4 on validation data and keeps the winner
- Evaluates the untouched test set exactly once
- Saves the trained weights to `iris_pytorch_weights.pt` and reloads them for one new prediction

## Results

- Validation accuracy: 0.933
- Final test accuracy: 0.933
- The only confusion was versicolor and virginica mixing once each way, which matches how close their petal measurements are

## How to run

1. Run `python -m pip install -r requirements.txt`
2. Keep `iris_demo.csv` in the same folder as the notebook
3. Run the cells top to bottom in Jupyter
