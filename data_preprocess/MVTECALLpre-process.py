# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:51:19 2019

@author: Anna
"""

from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from pycocotools import mask
import tensorflow as tf
import keras
import os
import shutil
from sklearn.model_selection import train_test_split



#==================load all the categories train================
annFile1='D:/dataset/d2s_annotations_v1.1/annotations/D2S_validation.json'
coco=COCO(annFile1)

cats = coco.loadCats(coco.getCatIds())
catIds=[cat['id'] for cat in cats]

imgids=[]
for i, value in enumerate(catIds):
    imgid=coco.getImgIds(catIds=catIds[i])
    imgids.extend(imgid)

imgids = list(set(imgids))

imgDict = coco.loadImgs(imgids)
img1 = pd.DataFrame.from_dict(imgDict)
data_list1=img1.filter(['file_name','id'])

X_train,X_val1 = train_test_split(data_list1,test_size=0.2, random_state=42,
                                 shuffle=True)
X_val,X_test = train_test_split(X_val1,test_size=0.5, random_state=42,
                                 shuffle=True)
img_id_train=X_train['id']
img_id_dev=X_val['id']
X_train=X_train['file_name']
X_val=X_val['file_name']
X_train.reset_index(drop=True,inplace=True)
X_val.reset_index(drop=True,inplace=True)

