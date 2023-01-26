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
