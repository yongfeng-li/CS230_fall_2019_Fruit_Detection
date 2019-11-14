# MASKRCNN

Reference github repo:
- https://github.com/matterport/Mask_RCNN

## Steps to replicate Yongfeng's result

1. git clone the reference github repo
1. place the folder of ```coco``` to the repo folder: ```samples/coco```
1. download pre-trained weights: ```mask_rcnn_coco.h5``` and put to root folder of refer repo
1. prepare D2S dataset
   - create folder of ```datasets```
   - put images and annotations of d2s as to the folder
        - d2s_dataset with subfolder of "annotations" and images"

## Command to run the code:

- Inference with pre-trained weights: 
```python3 coco_d2s.py evaluate --dataset='../../datasets/d2s_dataset' --model=coco --subset='D2S_validation'```

- Train model with pre-trained weights:
```python3 coco_d2s.py train --dataset='../../datasets/d2s_dataset' --model=coco --subset='D2S_validation'```


## Note: 
- create the virtual environment and run there
- need to install maskrcnn correctly
- need to install coco API correctly
