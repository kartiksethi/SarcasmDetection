# SarcasmDetection
A deep learning implementation to classify texts as sarcastic or not sarcastic.

Steps to be followed to make the code work:
* Install latest version of [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/install/)
* Install the other python libraries:numpy, pandas, scikit-learn, matplotlib, etc.
* **sarcasm_classifier.py** contains the code for training and testing the models
Command line args:
'''python
  --epochs: Number of sweeps over the dataset to train
  --lr: Learning Rate for the optimizer
  --optimizer: Type of optimizer
  --batchsize: Number of sequences to process at a time
  --model-type: Type of model to run from models.py
  --kfolds: Defines whether we want to do k-fold cross-validation or not
  --save: File path to save the final model
  --model: Give a file path of model to test
  --transfer-learning: File path to pre-trained glove embeddings
  --trained-embeddings: File path to word embeddings of words from sarcasmV2 dataset
  --data-path: File path to dataset file
  --model-summary: To enable or disable model summary
'''
