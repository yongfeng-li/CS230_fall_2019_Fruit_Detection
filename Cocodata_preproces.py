# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:47:02 2019

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

#========================Step 1: Prepare Coco dataset ======================
# 1) find classes
# 2) get image ids & annotations 
#===========annotation file======================
dataDir='D:/coco'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)

#==================load all the categories================
cats = coco.loadCats(coco.getCatIds())
coco_cat=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(coco_cat)))

coco_super = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(coco_super)))

# ====================get all images containing given categories, select one at random
catIds1= coco.getCatIds(catNms=["apple"]) #getImIds not working for multiple class
catIds2=coco.getCatIds(catNms=["banana"])
catIds3=coco.getCatIds(catNms=["orange"])
catIds4=coco.getCatIds(catNms=["broccoli"])
catIds5=coco.getCatIds(catNms=["carrot"])

imgIds1 = coco.getImgIds(catIds=catIds1 )
imgIds2 = coco.getImgIds(catIds=catIds2 )
imgIds3 = coco.getImgIds(catIds=catIds3 )
imgIds4 = coco.getImgIds(catIds=catIds4 )
imgIds5 = coco.getImgIds(catIds=catIds5 )

imgIds=imgIds1+imgIds2+imgIds3+imgIds4+imgIds5

imgDict = coco.loadImgs(imgIds)
img1 = pd.DataFrame.from_dict(imgDict)
data_list=img1['file_name']

X_train,X_val = train_test_split(data_list,test_size=0.3, random_state=42)
X_train.reset_index(drop=True,inplace=True)
X_val.reset_index(drop=True,inplace=True)
#=====================create folder for all fruit picutres===========

dest='D:/coco/train/'
src='D:/coco/images/'

full_lists=[]
for idx, val in enumerate(X_train):
    full_list=os.path.join(src, X_train[idx])
    full_lists.append(full_list)

for idx, val in enumerate(full_lists):
    file=full_lists[idx]
    shutil.copy(file, dest)
    
dest1='D:/coco/val/'

full_lists1=[]
for idx, val in enumerate(X_val):
    full_list=os.path.join(src, X_val[idx])
    full_lists1.append(full_list)

for idx, val in enumerate(full_lists1):
    file=full_lists1[idx]
    shutil.copy(file, dest1)
        




