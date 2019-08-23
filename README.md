# COCO-Caption

**TASK - IMAGE CAPTIONING**
The repository contains the code for training an image captioning model on MS COCO Dataset. For training purpose, Convolutional Neural Network and LSTM concepts are used.

**DEPENDENCIES**
- temsorflow: Tensorflow is a deep learning library developed by Google to create deep learning models.
- keras: Keras is a open-source deep learning library used to create and evaluate deep learning models.
- numpy: Numpy is python library used to perform high level mathematical functions on arrays and matrices.
- random: Random library is used to get a random number/ sample from a range or set of values.
- math: Math library is used to perform mathematical functions.
- os: OS library provides functioanlity to interact with the operating system.
- datetime: Datetime library supplies classes for manipulating dates and times in both simple and complex ways.

**FILES**
- datasets.py: This file loads the data from the disk.
- data_provider.py: This file provides data to the image captioning model.
- config.py: This file contains the hyperparameters value of the image captioning model.
- losses.py: This file contains the model losses.
- callbacks.py: This file conains various callbacks used during model training.
- metrics.py: This file contains the metrics of the model.
- model.py: This file builds the structure of the model before training.
- preprocessors.py: This file contains the Image and Caption preprocessors.
- training.ipynb: This python notebook contains the code to train the model.
