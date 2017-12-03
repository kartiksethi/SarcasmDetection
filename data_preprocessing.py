import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
random_seed = 99
np.random.seed(random_seed)


class SarcasmDataLoader:
    """
    ### Class used to load and process the dataset/word embeddings.
    """
    def __init__(self, max_len):
        # Data processing using Keras Tokenizer Class
        self.tokenizer = Tokenizer()
        self.max_len = max_len  # defines the max_len of each sentence seq

    """
    ### Function to load and process data
    """
    def loadData(self, path):
        print('-' * 100)
        sarcasm_data = pd.read_csv(path)

        # printing the data
        # print(sarcasm_data)

        # generating the labels (converting 'sarc' = 1 and 'notsarc' = 0)
        labels = sarcasm_data['Label']
        labels = np.array(labels)
        labels = labels == 'sarc'
        labels = labels.astype('float')

        # This list will store each sentence/sequence
        seq_matrix = []
        # Total number of Sequences
        # According to the dataset only 'Response Text' attribute corresponds
        # to sarcastic/non-sarcastic sentences
        number_of_sequences = len(sarcasm_data['Response Text'])
        for i in range(number_of_sequences):
            seq_matrix.append(sarcasm_data['Response Text'][i])

        # list of texts on which we will do the training
        self.tokenizer.fit_on_texts(seq_matrix)
        # converting the list of text into sequences
        encoded_seq_matrix = self.tokenizer.texts_to_sequences(seq_matrix)
        # calculating the vocabulary size
        vocab_size = len(self.tokenizer.word_index) + 1
        # padding the seq_matrix with zeros (adding zeros to sequences having
        # length < max_len)
        padded_encoded_seq_matrix = pad_sequences(
            encoded_seq_matrix, maxlen=self.max_len, padding='post')

        # dividing the data into train/test splits and shuffling the data
        rng = np.arange(number_of_sequences)
        np.random.shuffle(rng)

        # dividing into an 80/20 train/test split
        train_rng = rng[:int(0.8 * number_of_sequences)]
        test_rng = rng[int(0.8 * number_of_sequences):]
        train_seq = padded_encoded_seq_matrix[train_rng]
        train_labels = labels[train_rng]
        test_seq = padded_encoded_seq_matrix[test_rng]
        test_labels = labels[test_rng]

        return train_seq, train_labels, test_seq, test_labels, vocab_size

    """
    ### Function to load the pre-trained word-embeddings
    """
    def loadPreTrainedWordEmbeddings(self, vocab_size, glove_embeddings=True, embedding_path='glove.6B/glove.6B.300d.txt'):
        # Loading the pre-trained embeddings into memory
        # Load one of glove.6B.50d/100d/200d/300d.txt depending on the complexity of model
        # Or directly load the saved embeddings (with respect to this data)

        # Open the file containing the glove embeddings
        if glove_embeddings == True:
            print("Loading the GloVe embeddings")
            glove_encoding = {}
            f = open(embedding_path)
            for line in f:
                word_vector = line.split()
                glove_encoding[word_vector[0]] = np.array(
                    word_vector[1:], dtype='float32')
            f.close()
            print("GloVe Embeddings loaded")
            # Now finding the embedding for the words present in our data
            # use same embedding dim as per the GloVe file used (here 300d.txt)
            word_embeddings = np.zeros((vocab_size, 300))
            word_to_idx = self.tokenizer.word_index.items()
            count = 0
            for word, index in word_to_idx:
                word_embedding_vector = glove_encoding.get(word)
                if word_embedding_vector is not None:
                    count += 1
                    word_embeddings[index] = word_embedding_vector

            print("Found {} words out of {} vocabulary words".format(count, vocab_size))

            # saving the embeddings for this dataset so we needn't open the
            # glove file next time
            word_embeddings = np.array(word_embeddings)
            # Use this file with '--trained-embeddings' command line arg when training again
            np.save('word_embeddings_sarcasm_dataset.npy', word_embeddings)
            print("Saved word embeddings loaded")
            return word_embeddings, 300   # change this 300 accordingly

        # Loading the embeddings from an already saved embedding file
        else:
            word_embeddings = np.load(embedding_path)
            print("Saved word embeddings loaded")
            return word_embeddings, 300   # change this 300 accordingly
