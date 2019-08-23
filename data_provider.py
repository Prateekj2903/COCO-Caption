
# coding: utf-8

# In[1]:


import datasets
import config
import random
from math import ceil
import numpy as np
import os

import preprocessors


# In[2]:


class DataProvider():
    def __init__(self, best_caption=False, batch_size=None):
        
        self.active_config = config.Config()
        self.best_caption = best_caption
        self.batch_size = batch_size or self.active_config.batch_size
        
        self.dataset = datasets.COCODateset(best_caption)
        self.image_preprocessor = preprocessors.ImagePreprocessor()
        self.caption_preprocessor = preprocessors.CaptionPreprocessor()
        
        self.build()
    
    @property
    def vocabulary(self):
        return self.caption_preprocessor.vocabulary
    
    @property
    def vocabulary_size(self):
        return self.caption_preprocessor.vocabulary_size
    
    @property
    def training_results_dir(self):
        return self.dataset.training_results_dir
    
    @property
    def training_images_dir(self):
        return self.dataset.training_images_dir
    
    @property
    def validation_images_dir(self):
        return self.dataset.validation_images_dir
    
    @property
    def training_steps_per_epoch(self):
        return ceil(self.dataset.training_set_size/self.batch_size)
    
    @property
    def validation_steps_per_epoch(self):
        return ceil(self.dataset.validation_set_size/self.batch_size)
    
    def training_set_generator(self):
        for batch in self.batch_generator(self.dataset.training_set, is_training_set=True):
            yield batch
            
    def validation_set_generator(self):
        for batch in self.batch_generator(self.dataset.validation_set, is_training_set=False):
            yield batch
        
    def build(self):
        training_all_captions = self.dataset.training_all_captions
        self.caption_preprocessor.fit_captions(training_all_captions)
        
    def batch_generator(self, dataset, is_training_set):
        curr_data_batch = []
        while True:
            idxs = np.random.randint(low=0, high=len(dataset), size=self.batch_size)
            curr_data_batch = dataset[idxs]
            yield self.preprocess_batch(curr_data_batch, is_training_set)
    
    def preprocess_batch(self, data_batch, is_training_set):
        imgs_path = []
        captions = []
        
        img_dir = self.training_images_dir if is_training_set else self.validation_images_dir
        
        for data in data_batch:
            imgs_path.append(os.path.join(img_dir, data.filename))
            captions.append(data.caption)
        
        img_batch = self.image_preprocessor.preprocess_images(imgs_path, is_training_set)
        caption_batch = self.caption_preprocessor.encode_captions(captions, preprocess=True)
        
        captions_input, captions_output = caption_batch
        X, y = [img_batch, captions_input], captions_output
        
        return X, y


# In[5]:


# data_provider = DataProvider()


# In[6]:


# batch = next(data_provider.validation_set_generator())
# len(batch), batch[0][0].shape, batch[0][1].shape, batch[1].shape


# In[37]:


# data_provider.vocabulary_size


# In[38]:


# batch[0][1][0]


# In[39]:


# batch[1][0]


# In[40]:


# maxi = np.argmax(batch[1], axis=2)
# maxi[0]


# In[41]:


# t = data_provider.caption_preprocessor.tokenizer
# t.word_index['spreadeagled']
# t.index_word[27549]
# index_word = t.index_word
# # sorted(index_word, key=index_word.get)

# # t.word_index
# # index_word


# In[42]:


# for q in batch[0][1][0]:
#     print(t.index_word[q], end=' ')
# print()
# for q in maxi[0]:
#     print(t.index_word[q], end=' ')


# In[43]:


# b = batch[2][0]


# In[44]:


# t.index_word[27549]


# In[45]:


# b.caption, b.filename, b.image_id


# In[32]:


# tokenizer = Tokenizer()


# In[33]:


# tokenizer.sequences_to_matrix()


# In[34]:


# from keras.utils import to_categorical

# docs = ['Well done a',
# 'Good work a']
# # 'Great effort a',
# # 'nice work a',
# # 'Excellent! v a']
# # create the tokenizer
# t = Tokenizer()
# # fit the tokenizer on the documents
# t.fit_on_texts(docs)
# # summarize what was learned
# # print(t.word_counts)
# # print(t.document_count)
# # print(t.word_index)
# # print(t.word_docs)
# # integer encode documents

# print(t.texts_to_sequences(docs))

# print(len(t.word_counts))
# a = np.array(t.texts_to_sequences(docs))
# # print(a)
# a = np.expand_dims(a, -1)
# print(a)
# print()

# b = to_categorical(a)
# print(b)

# # c = np.array([[3, 4, 1, 0, 0, 0], [5, 2, 1, 0, 0, 0], [6, 7, 1, 0, 0, 0], [8, 2, 1, 0, 0, 0], [9, 10, 1, 0, 0, 0]])

# # c = np.expand_dims(c, -1)
# # print(a.shape)
# # print(a)
# # b = t.sequences_to_matrix(a.tolist())
# # print(b)
# # print()
# # print(b.shape)
# # encoded_docs = t.texts_to_matrix(docs, mode='count')
# # print(encoded_docs)


# In[35]:


# a = [1,2,3,4]
# b = np.array([1,2,3,4])
# print(a)
# print(b)


# In[36]:


# a = np.array([[1,2,3,4],[5,6,7,8,9]])
# max(len(i) for i in a)


# In[37]:


# a=np.expand_dims(a, -1)
# a.shape


# In[38]:


# a[:,:,1:]


# In[39]:


# seq = [[1,2,3,4,5,0,0]]


# In[40]:


# from math import ceil
# ceil(1)

