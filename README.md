# SarcasmDetection

A Bi-LSTM based attention model implementation to classify texts as sarcastic or not sarcastic.

Dataset used: [sarcasmV2 dataset](https://nlds.soe.ucsc.edu/sarcasm2)

### Steps to be followed to make the code work:
* Install latest version of [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/install/)
* Install the other python libraries: numpy, pandas, scikit-learn, matplotlib, etc.
* **data_preprocessing.py** contains the script to pre-process and load the data.
* All the different models are stored in **models.py**.
* **sarcasm_classifier.py** is the main file and it contains the code for training and testing the models.
* Command line arguments for sarcasm_classifier.py:
```
  [--epochs]: Number of sweeps over the dataset to train
  [--lr]: Learning Rate for the optimizer
  [--optimizer]: Type of optimizer
  [--batchsize]: Number of sequences to process at a time
  [--model-type]: Type of model to run from models.py
  [--kfolds]: Defines whether we want to do k-fold cross-validation or not
  [--save]: File path to save the final model
  [--model]: Give a file path of model to test
  [--transfer-learning]: File path to pre-trained glove embeddings
  [--trained-embeddings]: File path to word embeddings of words from sarcasmV2 dataset
  [--data-path]: File path to dataset file
  [--model-summary]: To enable or disable model summary
```

#### Note: Please download the [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) ([glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)) and store them in appropriate directory before proceeding with transfer learning. One can use their own pre-trained word-embeddings as well.

