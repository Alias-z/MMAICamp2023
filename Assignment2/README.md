# Using mmpretrain to train ResNet50 and classcify fruits

## Dataset
Created by 张子豪. Download link: https://pan.baidu.com/share/init?surl=YgoU1M_v7ridtXB9xxbA1Q, extration code: 52m9

## Related tutorials on mmpretrain
1. mmpretarin introduction: https://www.bilibili.com/video/BV1PN411y7C1/?spm_id_from=333.999.0.0&vd_source=2af21dbdecafe5884922ea5c29d9991c
2. mmpretarin code demo: https://www.bilibili.com/video/BV1Ju4y1Z7ZE/?spm_id_from=333.999.0.0


## Tasks
1. Install mmpretrain
```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc3'
mim install "mmpretrain[multimodal]>=1.0.0rc8"
```

2. config ResNet-50
```python
!mim search mmpretrain --model resnet
!mim download mmpretrain --config resnetv1c50_8xb32_in1k --dest Configs
```



3. split dataset into train and val
```python
import os
import shutil
import random
import numpy as np
from tqdm import tqdm

def data_split(images_dir, r_train=0.8):
    """split dataset into train and val with defined ratio"""
    random.seed(0); np.random.seed(0) # set seed 
    image_types = ['.jpg', '.jpeg', '.png', '.tif'] # supported image types
    file_names = sorted(os.listdir(images_dir), key=str.casefold) # obtain all file names
    file_names = [name for name in file_names if any(name.endswith(file_type) for file_type in image_types)] # filter for only image files
    
    train_size = int(len(file_names)*r_train) # training size
    validation_size = int(len(file_names)*(1-r_train)) # validation size
    
    file_names_shuffle = file_names.copy() # make a copy of list to prevent changes in place
    random.shuffle(file_names_shuffle) # random shuffle file names   
    train_names = file_names_shuffle[:train_size] # file names for training
    val_names = file_names_shuffle[train_size:train_size + validation_size] # file names for validation
    print(f'train size={train_size}, validation size={validation_size}')
    return train_names, val_names


def dataset_creator(data_root, r_train=0.8):
    """create train and val dataset"""
    data_root = os.path.normpath(data_root) # format string as path
    train_root = os.path.join(data_root, 'train') # train set root
    val_root = os.path.join(data_root, 'val') # validation set root

    subfolders = os.listdir(data_root) # get subfolder names under data root
    for subfolder in subfolders:
        subfolder_path = os.path.join(data_root, subfolder) # subfolder path
        train_names, val_names = data_split(subfolder_path, r_train=r_train) # file names for train and val
        destination_train_root = os.path.join(train_root, subfolder) # train class folder dir
        destination_val_root = os.path.join(val_root, subfolder) # val class folder dir
        os.makedirs(destination_train_root, exist_ok=True); os.makedirs(destination_val_root, exist_ok=True) # create the folders for the given class
        for name in train_names:
            source = os.path.join(subfolder_path, name) # source train image           
            destination = os.path.join(destination_train_root, name) # destination train image
            shutil.copy2(source, destination) # copy source and paste to destination
        for name in val_names:
            source = os.path.join(subfolder_path, name) # source validation image           
            destination = os.path.join(destination_val_root, name) # destination validation image
            shutil.copy2(source, destination) # copy source and paste to destination
    return None

dataset_creator(data_root='fruit30_train', r_train=0.8)
```

4. modify config file accordingly and train ResNet-50
```shell
mim train mmpretrain Configs//mmpretrain_fruit.py
```

## Results

Test on val set
```shell
mim test mmpretrain Models//ResNet50//mmpretrain_fruit.py --checkpoint \
       Models//ResNet50//best_accuracy_top1_epoch_156.pth --gpus 1 
```
```shell
Loads checkpoint by local backend from path: Models//ResNet50//best_accuracy_top1_epoch_156.pth
06/08 08:50:12 - mmengine - INFO - Load checkpoint from Models//ResNet50//best_accuracy_top1_epoch_156.pth
06/08 08:51:43 - mmengine - INFO - Epoch(test) [27/27]    accuracy/top1: 93.9394  accuracy/top5: 99.0676  data_time: 3.3162  time: 3.3822
Testing finished successfully.
```

## Inference on test images
```python
import cv2
from matplotlib import pyplot as plt
from mmpretrain import ImageClassificationInferencer as classifier

model = classifier(model='Models//ResNet50//mmpretrain_fruit.py', 
                   pretrained='Models//ResNet50//best_accuracy_top1_epoch_156.pth')
images = os.listdir('Test')
images = [os.path.join('Test', image) for image in images]
results = model(images, batch_size=4, show_dir='Results')

for idx, result in enumerate(results):
    print(f"predicted class {result['pred_class']}, score = {result['pred_score']}")
    plt.imshow(cv2.cvtColor(cv2.imread(images[idx]), cv2.COLOR_RGB2BGR))
    plt.show()
```

Inference on some fruits images I took from Coop Zurich

```python
predicted class 香蕉, score = 0.9999998807907104
```
![IMG_9838.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment2/Test/IMG_9838.jpg)

```python
predicted class 猕猴桃, score = 0.4718630611896515
```
![IMG_9840.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment2/Test/IMG_9840.jpg)

```python
predicted class 黄瓜, score = 0.6168965697288513
```
![IMG_9840.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment2/Test/IMG_9841.jpg)

```python
predicted class 柠檬, score = 0.5859736800193787
```
![IMG_9841.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment2/Test/IMG_9842.jpg)
