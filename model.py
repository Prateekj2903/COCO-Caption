
# coding: utf-8

# In[1]:


import config

from keras.applications import inception_resnet_v2
from keras.layers import Input, Dense, BatchNormalization, LSTM, GRU, Concatenate, RepeatVector, Embedding, Bidirectional, TimeDistributed, Flatten
from keras.models import Model
from losses import categorical_crossentropy_from_logits
from keras.optimizers import Adam
from metrics import categorical_accuracy_with_variable_timestep


import numpy as np


# In[2]:


class ImageCaptioningModel():
    
    def __init__(self, embedding_size=None, activation=None, regularizer=None, learning_rate=None, vocab_size=None, rnn_type=None, bidirectional_rnn=None, rnn_layers=None, rnn_output_size=None, word_vector_init=None, pretrained_embeddings=None):
        self.active_config = config.Config()
        self.embedding_size = embedding_size or self.active_config.embedding_size
        self.activation = activation or self.active_config.activation
        self.regularizer = regularizer or self.active_config.regularizer
        self.learning_rate = learning_rate or self.active_config.learning_rate
        self.vocab_size = vocab_size or self.active_config.vocab_size
        self.rnn_type = rnn_type or self.active_config.rnn_type
        self.bidirectional_rnn = bidirectional_rnn or self.active_config.bidirectional_rnn
        self.rnn_layers = rnn_layers or self.active_config.rnn_layers
        self.rnn_output_size = rnn_output_size or self.active_config.rnn_output_size
        self.word_vector_init = word_vector_init or self.active_config.word_vector_init
        self.use_pre_trained_word_embeddings = self.active_config.use_pre_trained_word_embeddings
        self.pretrained_embeddings = pretrained_embeddings
#         self.build()
        
    def build(self, vocabulary=None):
        image_input, image_embedding = self.get_image_embedding()
        word_input, word_embedding = self.get_word_embedding()
        
        seq_input = Concatenate(axis=1)([image_embedding, word_embedding])
        seq_output = self.build_sequence_model(seq_input)
        
        model = Model(inputs=[image_input, word_input], outputs=seq_output)
        
        model.compile(optimizer=Adam(lr=self.learning_rate, clipnorm=5.0),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        self.keras_model = model
    
    def get_image_embedding(self):
        #TODO Build model to calculate image embedding
        image_model = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=self.active_config.image_input_shape)
        for layer in image_model.layers:
            layer.trainable = False
            
        flatten = Flatten()(image_model.output)
        dense_output = Dense(units=self.embedding_size, activation=self.activation, kernel_regularizer = self.regularizer)(flatten)
        image_embedding = RepeatVector(1)(dense_output)
        image_input = image_model.input
        
        return image_input, image_embedding
        
    def get_word_embedding(self):        
        input_sentence = Input(shape=[None])
        
        if self.use_pre_trained_word_embeddings:
            embedding_layer = Embedding(input_dim=self.vocab_size+1, output_dim=self.embedding_size, trainable=False)
            embedding_layer.build((None,))
            embedding_layer.set_weights([self.pretrained_embeddings])
        else:
            embedding_layer = Embedding(input_dim=self.vocab_size+1, output_dim=self.embedding_size, embeddings_regularizer=self.regularizer)
        
        embedding = embedding_layer(input_sentence)
        return input_sentence, embedding
        
    def build_sequence_model(self, seq_input):
        #TODO Build sequence model
        RNN = LSTM if self.rnn_type=='lstm' else GRU
        
        def rnn_layer():
            rnn = RNN(units=self.rnn_output_size, return_sequences=True)
            
            if self.bidirectional_rnn:
                rnn = Bidirectional(rnn)
            
            return rnn
        
        prev_input = seq_input
        
        for i in range(self.rnn_layers):
            prev_input = BatchNormalization()(prev_input)
            rnn_output = rnn_layer()(prev_input)
            prev_input = rnn_output
        time_distributed_dense_output = TimeDistributed(Dense(units=self.vocab_size+1))(rnn_output)
        return time_distributed_dense_output


# In[3]:


# model = ImageCaptioningModel(vocab_size=1000)


# In[6]:


# model.build()


# In[7]:


# model.keras_model.summary()

