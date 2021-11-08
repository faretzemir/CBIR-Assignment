# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 05:53:07 2021

@author: eilha
"""

import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle

#%%

DIRECTORY = r'C:\Users\eilha\Desktop\DIVP Assignment\CBIR-Assignment\dataset\camel\dog.jpg'
CATEGORIES = ['camel', 'classic', 'comma', 'deer', 'device0', 'device3','device4','device8','dog', 'face', 'fish', 'flatfish',
          'fountain', 'frog', 'guitar', 'hammer', 'hat', 'horse', 'horseshoe', 'phone']


#%%

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        print(img_path)
        img_arr = cv2.imread(img_path)
        cv2.imshow('Image', img_arr)
        break
    
    #%%
    

img = cv2.imread(DIRECTORY)
plt.imshow(img)