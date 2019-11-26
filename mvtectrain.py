# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:16:13 2019

@author: Anna
"""

# Load the Drive helper and mount
#======================
#from google.colab import drive
#drive.mount('/content/drive')

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import zipfile
import urllib.request
import shutil
import pandas as pd
import tensorflow as tf

ROOT_DIR = os.path.abspath("D:/coco")

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import MaskRCNN
from sklearn.model_selection import train_test_split
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO


#============================= Root directory of the project
sys.path.append(os.path.join(ROOT_DIR, "D:/CS 2301/Mask_RCNN/mrcnn/")) 

# =================Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

#=================================define functions ================================

class FruitDataset(utils.Dataset):
    def load_fruit(self,subset):
        
        #annotation path
        annFile='D:/dataset/d2s_annotations_v1.1/annotations/D2S_validation.json'
        coco=COCO(annFile)  
        
        dataset_dir='D:/dataset/images'
        assert subset in ["train", "val"]
            
        image_dir = 'D:/dataset/images'
        
        #=====need to run the coco_preprocess.py code
        if subset == "train" : 
            image_ids=img_id_train
        else:
            image_ids=img_id_dev
            
        # Add classes
        class_ids = sorted(coco.getCatIds())
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])


        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], iscrowd=False)))
        
        
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(self.__class__).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(self.__class__).image_reference(self, image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
        
    
def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


#=================================TRAINING SETP========================================

dataset_train = FruitDataset()
dataset_train.load_fruit(subset="train")
dataset_train.prepare()


dataset_val = FruitDataset()
dataset_val.load_fruit(subset="val")
dataset_val.prepare()


    
class FruitConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "fruit"
    

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    LEARNING_RATE =0.002
    
    
    #these 2 needs to set to training set / batch size
    STEPS_PER_EPOCH=2880//2
    
    #VALIDATION_STEPS=114//2

    GPU_COUNT = 1
      
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 60  # Background + fruit
        
    #all following config to fit into 16G GPU
    BACKBONE = "resnet50"
    
    IMAGE_MIN_DIM = 512

    IMAGE_MAX_DIM = 512

    
    #VALIDATION_STEPS = 5


config = FruitConfig()
config.display()

model = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='all')




#======================================Evaluate=====================

#=====================random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))


#================result
results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())



def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):

    """Runs official COCO evaluation.

    dataset: A Dataset object with valiadtion data

    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation

    limit: if not 0, it's the number of images to use for evaluation

    """
    # Pick COCO images from the dataset

    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.

    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]
    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):

        # Load image

        image = dataset.load_image(image_id)



        # Run detection

        t = time.time()

        r = model.detect([image], verbose=0)[0]

        t_prediction += (time.time() - t)



        # Convert results to COCO format

        # Cast masks to uint8 because COCO tools errors out on bool

        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],

                                           r["rois"], r["class_ids"],

                                           r["scores"],

                                           r["masks"].astype(np.uint8))

        results.extend(image_results)

    # Load results. This modifies results with additional attributes.

    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(

        t_prediction, t_prediction / len(image_ids)))

    print("Total time: ", time.time() - t_start)
   
    
evaluate_coco(model, dataset_val, coco, "bbox",limit=360)

