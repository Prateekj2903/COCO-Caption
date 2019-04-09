
# coding: utf-8

# In[ ]:


class Config():
    def __init__(self):
        self.clean_descriptions = True
        self.batch_size = 32
        self.embedding_size = 512
        self.activation = 'relu'
        self.regularizer = 'l2'
        self.vocab_size = None
        self.rnn_type = 'lstm'
        self.bidirectional_rnn = False
        self.rnn_layers = 2
        self.rnn_output_size=512

# In[ ]:


active_config = Config()



# dataset_name='flickr8k',
#                       epochs=None,
#                       time_limit=timedelta(hours=24),
#                       batch_size=32,
#                       reduce_lr_factor=0.7,
#                       reduce_lr_patience=4,
#                       early_stopping_patience=sys.maxsize,
#                       lemmatize_caption=True,
#                       rare_words_handling='discard', ---------- Done
#                       words_min_occur=5,
#                       learning_rate=0.001,
#                       vocab_size=None,
#                       embedding_size=512,
#                       rnn_output_size=512,
#                       dropout_rate=0.3,
#                       bidirectional_rnn=False,
#                       rnn_type='lstm',
#                       rnn_layers=1,
#                       l1_reg=0.0,
#                       l2_reg=0.0,
#                       initializer='vinyals_uniform',
#                       word_vector_init=None,
#                       image_augmentation=False