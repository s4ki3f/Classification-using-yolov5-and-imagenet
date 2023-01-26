# Classification-using-yolov5-and-imagenet
# YOLOv5 Classification Notebook

YOLOv5 supports classification tasks too. YOLOv5 is maintained by [Ultralytics](https://github.com/ultralytics/yolov5).

This notebook covers:

*   Inference with out-of-the-box YOLOv5 classification on ImageNet
*  Training YOLOv5 classification on custom data



This notebook was created with Google Colab. [Click here](https://colab.research.google.com/github/s4ki3f/Lightning-Series/blob/main/YOLOv5_Classification_Tutorial.ipynb#scrollTo=5GYQX3of4QiW) to run it.
<br>
<div>
  <a href="https://colab.research.google.com/github/s4ki3f/Lightning-Series/blob/main/YOLOv5_Classification_Tutorial.ipynb#scrollTo=5GYQX3of4QiW"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>
<br>

# Setup

Pull in respective libraries to prepare the notebook environment.
```bash
!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
%pip install -qr requirements.txt  # install

import torch
import utils
display = utils.notebook_init()  # checks
```
# 1. Infer on ImageNet

To demonstrate YOLOv5 classification, we'll leverage an already trained model. In this case, we'll download the ImageNet trained models pretrained on ImageNet using YOLOv5 Utils.

```python
from utils.downloads import attempt_download

p5 = ['n', 's', 'm', 'l', 'x']  # P5 models
cls = [f'{x}-cls' for x in p5]  # classification models

for x in cls:
    attempt_download(f'weights/yolov5{x}.pt')
```
## 2. Optional Validatation to check imagenet models

Use the `classify/val.py` script to run validation for the model. This will show us the model's performance on each class.

First, we need to download ImageNet.

```bash
!bash data/scripts/get_imagenet.sh --val
!python classify/val.py --weights ./weigths/yolov5s-cls.pt --data ../datasets/imagenet
```


## 3. Load  Dataset

Next, we'll export our dataset into the right directory structure for training YOLOv5 classification to load into this notebook. Select the `Export` button at the top of the version page, `Folder Structure` type, and `show download code`.

The ensures all our directories are in the right format:

```
dataset
├── train
│   ├── class-one
│   │   ├── IMG_123.jpg
│   └── class-two
│       ├── IMG_456.jpg
├── valid
│   ├── class-one
│   │   ├── IMG_789.jpg
│   └── class-two
│       ├── IMG_101.jpg
├── test
│   ├── class-one
│   │   ├── IMG_121.jpg
│   └── class-two
│       ├── IMG_341.jpg
```



#Dataset used here are from, we can use roboflow lib to download dataset easily

```bash
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="deO5u1tP8oXg2HyoSRT4")
project = rf.workspace("tdc").project("cow-identification")
dataset = project.version(1).download("folder")
```

## 4. Train On Custom Data 
Here, we use the DATASET_NAME environment variable to pass our dataset to the `--data` parameter.

Note: we're training for 100 epochs here. We're also starting training from the pretrained weights. Larger datasets will likely benefit from longer training. 

##And we will be need of traning 5-6 times to tune to the hyperperameters to find the best perameter for best training output

```python
!python classify/train.py --model yolov5x-cls.pt --data $DATASET_NAME --epochs 100 --img 128 
```

## 5. Validate Your Custom Model

Repeat step 2 from above to test and validate your custom model.

```python
!python classify/val.py --weights runs/train-cls/exp/weights/best.pt --data ../datasets/$DATASET_NAME
```
