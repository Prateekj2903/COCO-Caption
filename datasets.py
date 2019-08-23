
# coding: utf-8

# In[1]:


from pycocotools.coco import COCO
import os
import numpy as np


# In[5]:


# TODO build function to create dictionary of all captions for an image
class Image():
    def __init__(self, image_id, caption, filename):
        self.image_id = image_id
        self.caption = caption
        self.filename = filename

class COCODateset(object):
    #Load Data from disk
    DATASET_NAME = 'COCO'

    #WINDOWS
    DATASET_DIR = 'D:\\Datasets\\COCO'
    TRAINING_RESULTS_DIR = 'D:\\img_captioning\\training_results'

    #UBUNTU
    # DATASET_DIR = '/media/prateek/New Volume/Datasets/COCO'
    # TRAINING_RESULTS_DIR = '/media/prateek/New Volume/img_captioning/training_results'

    #COLAB
    # DATASET_DIR = '/content/COCO'
    # TRAINING_RESULTS_DIR = '/content/drive/My Drive/COCO/training_results'

    ANNOTAIONS_DIR = os.path.join(DATASET_DIR,'annotations')
    
    COCO_TRAIN_ANNOTATION_FILENAME = 'instances_train2017.json'
    COCO_TRAIN_CAPTION_ANNOTAION_FILENAME = 'captions_train2017.json'
    
    COCO_VAL_ANNOTATION_FILENAME = 'instances_val2017.json'
    COCO_VAL_CAPTION_ANNOTAION_FILENAME = 'captions_val2017.json'
    
    
    @property
    def dataset_dir(self):
        return self.DATASET_DIR
    @property
    def training_results_dir(self):
        return self.TRAINING_RESULTS_DIR
    @property
    def training_images_dir(self):
        return os.path.join(self.dataset_dir, 'train2017')
    @property
    def validation_images_dir(self):
        return os.path.join(self.dataset_dir, 'val2017')
    @property
    def training_set_size(self):
        return len(self.training_set)
    @property
    def training_all_captions_size(self):
        return len(self.training_all_captions)
    @property
    def validation_set_size(self):
        return len(self.validation_set)
    @property
    def validation_all_captions_size(self):
        return len(self.validation_all_captions)
    
    def __init__(self, single_caption=False):
        self.single_caption = single_caption
        self.build_data()
#         self.training_set_size = len(self.training_set)
        
#         self.validation_set_size = len(self.validation_set)
#         self.test_set_size = len(self.test_set)
    
    def load_annotations(self, coco_ann_filepath, coco_caption_ann_filepath):
        coco_ann = COCO(coco_ann_filepath)
        coco_caption_ann = COCO(coco_caption_ann_filepath)
        return coco_ann, coco_caption_ann
        
    def load_set(self, coco_ann_filepath, coco_caption_ann_filepath):
        
        #TODO if best_caption then save ony one caption per image
        coco_ann, coco_caption_ann = self.load_annotations(coco_ann_filepath, coco_caption_ann_filepath)
        img_ids = coco_ann.getImgIds()
        annIds = coco_caption_ann.getAnnIds(img_ids)
        annotations = coco_caption_ann.loadAnns(annIds)
#         dataset = defaultdict(list)
        is_image_id_done = dict()
        dataset = []
        all_captions = []
        for ann in annotations:
            image_id = ann['image_id']
            image_caption = ann['caption']
            image_filename = coco_ann.loadImgs([image_id])[0]['file_name']
            all_captions.append(ann['caption'])
            if self.single_caption and image_id in is_image_id_done.keys():
                continue
            else:
                dataset.append(Image(image_id, image_caption, image_filename))
                is_image_id_done[image_id] = True
        return np.array(dataset), all_captions
    def build_data(self):
        print("Loading Training Data")
        self.training_set, self.training_all_captions = self.load_set(os.path.join(self.ANNOTAIONS_DIR, self.COCO_TRAIN_ANNOTATION_FILENAME),
                                os.path.join(self.ANNOTAIONS_DIR, self.COCO_TRAIN_CAPTION_ANNOTAION_FILENAME))
        print("\nLoading Validation Data")
        self.validation_set, self.validation_all_captions = self.load_set(os.path.join(self.ANNOTAIONS_DIR, self.COCO_VAL_ANNOTATION_FILENAME),
                                os.path.join(self.ANNOTAIONS_DIR, self.COCO_VAL_CAPTION_ANNOTAION_FILENAME))
#         self.test_set = 


# In[6]:


# c = COCODateset(single_caption=True)


# In[8]:


# print(len(c.training_set), len(c.training_all_captions))
# print(len(c.validation_set), len(c.validation_all_captions))


# In[14]:


# c.training_set[0].image_id, c.training_set[0].caption, c.training_set[0].filename


# In[15]:


# c.training_set[1].image_id, c.training_set[1].caption, c.training_set[1].filename

