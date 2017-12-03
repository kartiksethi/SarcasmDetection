from __future__ import print_function, division
from keras.layers import Dense, Flatten, Dropout, Activation, Conv1D, Permute, RepeatVector, multiply, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, Model
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Input, Lambda
import tensorflow as tf
from keras import backend as K



class SarcasmModel:

    def __init__(self, model_type, hidden_units, embedding_dim, vocab_size, max_len, pre_trained_embedding=False, embedding_weights=None):
        self.model_type = model_type
        self.hidden_units = hidden_units
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pre_trained_embedding = pre_trained_embedding  # using pre-trained word embeddings or not
        self.embedding_weights = embedding_weights

    def loadModel(self):
    	# Decides whether to use pre-trained embedding weights or not
		input = Input(shape=(self.max_len,))
		if self.pre_trained_embedding == True:
			embeddings = Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len, weights=[self.embedding_weights], trainable=False)(input)
		else:
			embeddings = Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len)(input)

		# Selecting the model based on the type
		if self.model_type == 1:  # basic neural network
			print('-' * 100)
			print("Model Selected: Basic neural network")
			print('-' * 100)
			lstm_output = Flatten()(embeddings)
			final_output = Dense(1, activation='sigmoid')(lstm_output)

		elif self.model_type == 2:  # LSTM based network
			print('-' * 100)
			print("Model Selected: LSTM based network")
			print('-' * 100)
			lstm_output = LSTM(self.hidden_units)(embeddings)
			lstm_output = Dense(256, activation ='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(lstm_output)
			lstm_output = Dropout(0.3)(lstm_output)
			lstm_output = Dense(128, activation ='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(lstm_output)
			lstm_output = Dropout(0.3)(lstm_output)
			final_output = Dense(1, activation='sigmoid')(lstm_output)

		elif self.model_type == 3:  # Bidirectional LSTM without attention
			print('-' * 100)
			print("Model Selected: Bidirectional LSTM without attention")
			print('-' * 100)
			lstm_output = Bidirectional(LSTM(self.hidden_units))(embeddings)
			lstm_output = Dense(256, activation ='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(lstm_output)
			lstm_output = Dropout(0.3)(lstm_output)
			lstm_output = Dense(128, activation ='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(lstm_output)
			lstm_output = Dropout(0.3)(lstm_output)
			final_output = Dense(1, activation='sigmoid')(lstm_output)

		elif self.model_type == 4:  # Bidirectional LSTM with attention
			print('-' * 100)
			print("Model Selected: Bidirectional LSTM with attention")
			print('-' * 100)
			lstm_output = Bidirectional(LSTM(self.hidden_units, return_sequences=True), merge_mode='ave')(embeddings)

			# calculating the attention coefficient for each hidden state
			attention_vector = Dense(1, activation='tanh')(lstm_output)
			attention_vector = Flatten()(attention_vector)
			attention_vector = Activation('softmax')(attention_vector)
			attention_vector = RepeatVector(self.hidden_units)(attention_vector)
			attention_vector = Permute([2, 1])(attention_vector)

			# Multiplying the hidden states with the attention coefficients and
			# finding the weighted average
			final_output = multiply([lstm_output, attention_vector])
			final_output = Lambda(lambda xin: K.sum(
			    xin, axis=-2), output_shape=(self.hidden_units,))(final_output)

			# passing the above weighted vector representation through single Dense
			# layer for classification
			final_output = Dropout(0.5)(final_output)
			final_output = Dense(256, activation ='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(final_output)
			lstm_output = Dropout(0.3)(final_output)
			final_output = Dense(128, activation ='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(final_output)
			final_output = Dense(1, activation='sigmoid')(final_output)

		elif self.model_type == 5: # CNN-Bidirectional LSTM with attention
			print('-' * 100)
			print("Model Selected: CNN-Bidirectional LSTM with attention")
			print('-' * 100)
			# Hyper parameters for 1D Conv layer
			filters = 32
			kernel_size = 5
			embeddings = Dropout(0.3)(embeddings)
			conv_output = Conv1D(filters, kernel_size, activation='relu')(embeddings)
			lstm_output = Bidirectional(LSTM(self.hidden_units, return_sequences=True), merge_mode='ave')(conv_output)

			# calculating the attention coefficient for each hidden state
			attention_vector = Dense(1, activation='tanh')(lstm_output)
			attention_vector = Flatten()(attention_vector)
			attention_vector = Activation('softmax')(attention_vector)
			attention_vector = RepeatVector(self.hidden_units)(attention_vector)
			attention_vector = Permute([2, 1])(attention_vector)

			# Multiplying the hidden states with the attention coefficients and
			# finding the weighted average
			final_output = multiply([lstm_output, attention_vector])
			final_output = Lambda(lambda xin: K.sum(
			    xin, axis=-2), output_shape=(self.hidden_units,))(final_output)

			# passing the above weighted vector representation through single Dense
			# layer for classification
			final_output = Dropout(0.5)(final_output)
			final_output = Dense(256, activation ='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(final_output)
			lstm_output = Dropout(0.3)(final_output)
			final_output = Dense(128, activation ='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(final_output)
			final_output = Dense(1, activation='sigmoid')(final_output)

		model = Model(inputs=input, outputs=final_output)
		return model


