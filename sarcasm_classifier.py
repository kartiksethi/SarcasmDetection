from __future__ import print_function, division
from keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
import matplotlib.pyplot as plt
import numpy as np
random_seed = 99
np.random.seed(random_seed)
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn
from models import SarcasmModel
from data_preprocessing import SarcasmDataLoader
import argparse


if __name__ == '__main__':
    # Defining the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', type=float, default=1e-03,
                        help='Learning Rate for the optimizer')
    parser.add_argument('--optimizer', type=str, default="Adam",
                        help='Type of optimizer')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='Number of sequences to process at a time')
    parser.add_argument('--model-type', type=int, default=5,
                        help='Type of model to run from models.py')
    parser.add_argument('--kfolds', type=bool, default=False,
                        help='Defines whether we want to do k-fold cross-validation or not')
    parser.add_argument('--save', type=str,  default='model.h5',
                        help='File path to save the final model')
    parser.add_argument('--model', default=None,
                        help='Give a file path of model to test')
    # give the name of the file path containing your pre-trained word-embeddings
    parser.add_argument('--transfer-learning', type=str, default='../glove.6B/glove.6B.300d.txt',
                        help='File path to pre-trained glove embeddings')
    parser.add_argument('--trained-embeddings', default=None,
                        help='File path to word embeddings of words from sarcasmV2 dataset')
    parser.add_argument('--data-path', default='sarcasm_v2.csv',
                        help='File path to dataset file')
    parser.add_argument('--model-summary', type=bool, default=False,
                        help='To enable or disable model summary')

    args = parser.parse_args()
    print("Args: %s" % args)

    # defining the various hyperparameters for the model
    hidden_units = 128  # hidden units for the LSTM
    # the max length of sentence sequence (longer sentences will get truncated
    # to this length)
    max_len = 50

    # Data preprocessing and loading
    # creating an object of the SarcasmDataLoader class
    data_loader_object = SarcasmDataLoader(max_len)
    x_train, y_train, x_test, y_test, vocab_size = data_loader_object.loadData(
        path=args.data_path)
    print("x_train, y_train shape:{}, {}".format(x_train.shape, y_train.shape))
    print("x_test, y_test shape:{}, {}".format(x_test.shape, y_test.shape))
    print("Vocab Size:{} words".format(vocab_size))
    print('-' * 100)

    # Loading pre-trained word embeddings
    if args.transfer_learning != "None":  # using pre-trained word embeddings
        if args.trained_embeddings != None:
            embedding_weights, embedding_dim = data_loader_object.loadPreTrainedWordEmbeddings(
                vocab_size, embedding_path=args.trained_embeddings, glove_embeddings=False)
        else:
            embedding_weights, embedding_dim = data_loader_object.loadPreTrainedWordEmbeddings(
                vocab_size, embedding_path=args.transfer_learning)
    else:
        embedding_dim = 80  # manually set the value, if not doing transfer learning

    # Load already trained model
    if args.model != None:
        model = model.load(args.model)
    else:
        # Retrieving the type of model given by the user
        # Creating an object of the SarcasmModel class
        if args.transfer_learning != "None":
            sarcasm_model_object = SarcasmModel(
                args.model_type, hidden_units, embedding_dim, vocab_size, max_len, True, embedding_weights)
        else:
            sarcasm_model_object = SarcasmModel(
                args.model_type, hidden_units, embedding_dim, vocab_size, max_len)

        model = sarcasm_model_object.loadModel()

        if args.optimizer == "Adam":
            opt = Adam(lr=args.lr)
        elif args.optimizer == "SGD":
            opt = SGD(lr=args.lr)

        #  Compile the model
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

        if args.model_summary == True:
            # Print model summary
            print(model.summary())

        # Training the model
        if args.kfolds == False:  # training without k-fold cross-validation
            plot_data = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batchsize,
                      verbose=1, shuffle=True, validation_split=0.1)
        else:
            # first argument refers to the number of folds in the train dataset
            kfold = StratifiedShuffleSplit(2, test_size=0.5, random_state=0)
            cvscores = []

            for train_index, test_index in kfold.split(x_train, y_train):
                # Training the model
                plot_data = model.fit(x_train[train_index], y_train[
                          train_index], epochs=args.epochs, verbose=1, batch_size=args.batchsize, validation_split=0.1)
                # Evaluate the model on validation data
                loss, accuracy = model.evaluate(
                    x_train[test_index], y_train[test_index], verbose=1)
                print()
                print("Validation Loss:{} \tValidation Accuracy:{}".format(loss, accuracy))
                cvscores.append(accuracy * 100)

            print("Validation Accuracy:%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

        # Evaluate the model on test set
        loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
        print('Train Loss:{} \tTrain Accuracy:{}'.format(loss, accuracy * 100))

        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print('Test Loss:{} \tTest Accuracy:{}'.format(loss, accuracy * 100))

        # Plotting the results
        # print(plot_data.history.keys())  # gives the different metrics stored during training

        print("The graphs are displayed")
        # Plotting the graph for accuracy
        plt.plot(plot_data.history['acc'], 'r-')
        plt.plot(plot_data.history['val_acc'], 'b-')
        plt.xlabel('# of Epochs')
        plt.ylabel('Accuracy')
        plt.title("Accuracy vs Epochs")
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

        # Plotting the graph for loss
        plt.plot(plot_data.history['loss'], 'r-')
        plt.plot(plot_data.history['val_loss'], 'b-')
        plt.xlabel('# of Epochs')
        plt.title("Loss vs Epochs")
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

