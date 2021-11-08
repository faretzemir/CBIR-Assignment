# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 00:29:33 2021

@author: eilha
"""

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np


DATADIR = r"C:\Users\eilha\Desktop\DIVP Assignment\CBIR-Assignment\dataset"
CATEGORIES = ['camel', 'classic', 'comma', 'deer', 'device0', 'device3','device4','device8','dog', 'face', 'fish', 'flatfish',
          'fountain', 'frog', 'guitar', 'hammer', 'hat', 'horse', 'horseshoe', 'phone']

#%%
#Loading data

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        cv2.imshow("babi", img_array)
        plt.show()
        break
    break

#%%
# Fetch training and validation data

#Now we can easily fetch our train and validation data.
#train = get_data(train_dir)
#al = get_data(test_dir)#