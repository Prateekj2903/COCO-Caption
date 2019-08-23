
# coding: utf-8

# In[1]:


import config
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence as keras_seq
from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications import inception_resnet_v2
import numpy as np
import os


# In[ ]:


class ImagePreprocessor():
    IMAGE_SIZE = (299, 299, 3)
    
    def __init__(self, image_augmentation=None):
        
        self.active_config = config.Config()
        self.image_data_generator = ImageDataGenerator()
        self.image_augmentation = image_augmentation or self.active_config.image_augmentation
        
    def preprocess_images(self, imgs_path, random_transform=True):
        img_batch = [self.preprocess_single_image(img_path, random_transform) for img_path in imgs_path]
        return np.array(img_batch)
         
    def preprocess_single_image(self, img_path, random_transform):
        img = load_img(img_path, target_size=self.IMAGE_SIZE)
        img = img_to_array(img)
        
        if self.image_augmentation and random_transform:
            img = self.image_data_generator.random_transform(img)
        img = inception_resnet_v2.preprocess_input(img)
        return img


# In[ ]:


class CaptionPreprocessor():
    EOS_TOKEN = 'xeosx'

    #WINDOWS
    GLOVE_DIR = "D:\\Datasets\\Glove.6B\\glove.6B.300d.txt"

    #UBUNTU
    # GLOVE_DIR = "/media/prateek/New Volume/Datasets/Glove.6B/glove.6B.300d.txt"

    #COLAB
    # GLOVE_DIR = "/content/glove.6B.300d.txt"

    def __init__(self, clean_descriptions=None, embedding_size=None):
        self.active_config = config.Config()
        self.tokenizer = Tokenizer()
        self.clean_descriptions = clean_descriptions or self.active_config.clean_descriptions
        self.use_pre_trained_word_embeddings = self.active_config.use_pre_trained_word_embeddings
        self.embedding_size = embedding_size or self.active_config.embedding_size
        
    @property
    def vocabulary(self):
        word_index = self.tokenizer.word_index
        return sorted(word_index, key=word_index.get)
    
    @property
    def vocabulary_size(self):
        return len(self.tokenizer.index_word.keys())
        
    def fit_captions(self, captions_list):
        if self.clean_descriptions:
            captions_list = self.description_cleaner(captions_list)
        captions_list = self.eos_adder(captions_list)
        self.tokenizer.fit_on_texts(captions_list)
        self.max_len = self.get_max_len(captions_list)
        if self.use_pre_trained_word_embeddings:
            self.calculate_word_embeddings()

    def get_max_len(self, captions_list):
        return max(len(caption_sequence) for caption_sequence in self.tokenizer.texts_to_sequences(captions_list))
        
    def calculate_word_embeddings(self):
        self.embedding_index = self.load_pretrained_word_embeddings()
        self.embedding_matrix = np.zeros(shape=(self.vocabulary_size+1, self.embedding_size), dtype='float32')
        
        for word, index in self.tokenizer.word_index.items():
            try:
                self.embedding_matrix[index, :] = self.embedding_index[word]
            except Exception as e:
                pass
        
    def load_pretrained_word_embeddings(self):
        embeddings_index = {}
        f = open(self.GLOVE_DIR, 'rb')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word.decode('utf-8')] = coefs
        f.close()
        return embeddings_index
    
    def encode_captions(self, captions, preprocess=False):
        captions = self.eos_adder(captions)
        encoded_captions = self.tokenizer.texts_to_sequences(captions)
        if preprocess:
            return self.preprocess_batch(encoded_captions)
        else:
            return encoded_captions
    
    def preprocess_batch(self, captions_list):
        captions = keras_seq.pad_sequences(captions_list, maxlen=self.max_len, padding='post')
        
        captions_extended1 = keras_seq.pad_sequences(captions, maxlen=self.max_len+1, padding='post')
        captions_extended1 = np.expand_dims(captions_extended1, -1)
        captions_one_hot = to_categorical(captions_extended1, num_classes=self.vocabulary_size+1)
        
#         captions_decreased = captions.copy()
#         captions_decreased[captions_decreased > 0] -= 1
#         captions_one_hot_shifted = captions_one_hot[:,:,1:]
                
        captions_input = captions
        captions_output = captions_one_hot
        return captions_input, captions_output
        
    def description_cleaner(self, captions_list):
        # TODO clean descriptions
        return captions_list
        
    def eos_adder(self, captions_list):
        return [caption + ' ' + self.EOS_TOKEN for caption in captions_list]

