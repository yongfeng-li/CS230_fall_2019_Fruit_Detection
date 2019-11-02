# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 10:51:35 2019

@author: Anna
"""


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
ROOT_DIR = os.path.abspath("D:/coco")

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import MaskRCNN
from sklearn.model_selection import train_test_split
# =================Import COCO config
# Root directory of the project
sys.path.append(os.path.join(ROOT_DIR, "D:/CS 2301/Mask_RCNN/mrcnn/")) 
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
import coco
%matplotlib inline

# =================Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "train")

subset='train'
dataset_dir='D:\coco'
dataset_dir = os.path.join(dataset_dir, subset)

# =================Step 3:data procss==============    
############################################################
#  Defind pre-process
############################################################
class FruitDataset(utils.Dataset):
    def load_fruit(self,subset):
        # Add classes
        self.add_class("fruit", 50, "orange")
        self.add_class("fruit", 51, "broccoli")
        self.add_class("fruit", 47, "banana")
        self.add_class("fruit", 48, "apple")
        self.add_class("fruit", 52, "carrot")
        
        #annotation path
        annFile='D:/coco/annotations/instances_train2014.json'
        coco=COCO(annFile)  
        
        dataset_dir='D:\coco'
        assert subset in ["train", "val"]
        image_dir = os.path.join(dataset_dir, subset)
        
        #=====need to run the coco_preprocess.py code
        image_ids=imgIds
        
        # Add classes
        self.add_class("fruit", 50, "orange")
        self.add_class("fruit", 51, "broccoli")
        self.add_class("fruit", 47, "banana")
        self.add_class("fruit", 48, "apple")
        self.add_class("fruit", 52, "carrot")

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

# =================Step 4:model training==============    
############################################################
#  Defind Model Training 
############################################################

dataset_train = FruitDataset()
dataset_train.load_fruit("train")
dataset_train.prepare()

dataset_val = FruitDataset()
dataset_val.load_fruit("val")
dataset_val.prepare()

# =================Step 5:Rewrite the config file==============
class FruitConfig(coco.Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "fruit"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

config = FruitConfig()
config.display()

model = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)
model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')
 
    


