# Using mmseg to segment watermelon

## Dataset
Created by 张子豪. Download link: https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/watermelon/Watermelon87_Semantic_Seg_Mask.zip

## Related tutorials on mmseg
1. mmseg introduction: https://www.bilibili.com/video/BV1gV4y1m74P/?spm_id_from=333.999.0.0&vd_source=2af21dbdecafe5884922ea5c29d9991c
2. mmseg code demo: https://www.bilibili.com/video/BV1uh411T73q/?spm_id_from=333.999.0.0&vd_source=2af21dbdecafe5884922ea5c29d9991c


## Tasks
1. Install mmseg
```python
!pip install -U openmim
!mim install mmengine
!mim install 'mmcv>=2.0.0rc3'
!mim install "mmdet>=3.0.0rc6"

#!rm -rf mmsegmentation
!git clone https://github.com/open-mmlab/mmsegmentation.git
%cd mmsegmentation
!pip install -v -e .
%cd ..
```

2. config PSPNet
```python
!mim search mmsegmentation --model pspnet
!mim download mmsegmentation --config pspnet_r101-d8_4xb4-80k_ade20k-512x512 --dest Configs
```

3. modify config file accordingly and create WatermelonDataset 
```shell
import os
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmseg.utils import register_all_modules

classes = ('background', 'red', 'green', 'white', 'seed-black', 'seed-white')
palette = [[0 , 0, 0], [255, 0, 0], [0, 255, 0], [255, 255, 255], [255, 255, 0], [255, 0, 255]]

@DATASETS.register_module()
class WatermelonDataset(BaseSegDataset):
  METAINFO = dict(classes=classes, palette=palette)
  def __init__(self, 
    img_suffix='.jpg',
    seg_map_suffix='.png',
    reduce_zero_label=False,
    **kwargs):
    super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
```



4. modify \_\_init_\_\.py to register WatermelonDataset
```shell
# Copyright (c) OpenMMLab. All rights reserved.
# yapf: disable
from .ade import ADE20KDataset
from .basesegdataset import BaseSegDataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .coco_stuff import COCOStuffDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import MultiImageMixDataset
from .decathlon import DecathlonDataset
from .drive import DRIVEDataset
from .hrf import HRFDataset
from .isaid import iSAIDDataset
from .isprs import ISPRSDataset
from .lip import LIPDataset
from .loveda import LoveDADataset
from .mapillary import MapillaryDataset_v1, MapillaryDataset_v2
from .night_driving import NightDrivingDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .potsdam import PotsdamDataset
from .refuge import REFUGEDataset
from .stare import STAREDataset
from .synapse import SynapseDataset
from .watermelonDataset import WatermelonDataset
# yapf: disable
from .transforms import (CLAHE, AdjustGamma, BioMedical3DPad,
                         BioMedical3DRandomCrop, BioMedical3DRandomFlip,
                         BioMedicalGaussianBlur, BioMedicalGaussianNoise,
                         BioMedicalRandomGamma, GenerateEdge, LoadAnnotations,
                         LoadBiomedicalAnnotation, LoadBiomedicalData,
                         LoadBiomedicalImageFromFile, LoadImageFromNDArray,
                         PackSegInputs, PhotoMetricDistortion, RandomCrop,
                         RandomCutOut, RandomMosaic, RandomRotate,
                         RandomRotFlip, Rerange, ResizeShortestEdge,
                         ResizeToMultiple, RGB2Gray, SegRescale)
from .voc import PascalVOCDataset

# yapf: enable
__all__ = [
    'BaseSegDataset', 'BioMedical3DRandomCrop', 'BioMedical3DRandomFlip',
    'CityscapesDataset', 'PascalVOCDataset', 'ADE20KDataset',
    'PascalContextDataset', 'PascalContextDataset59', 'ChaseDB1Dataset',
    'DRIVEDataset', 'HRFDataset', 'STAREDataset', 'DarkZurichDataset',
    'NightDrivingDataset', 'COCOStuffDataset', 'LoveDADataset',
    'MultiImageMixDataset', 'iSAIDDataset', 'ISPRSDataset', 'PotsdamDataset',
    'LoadAnnotations', 'RandomCrop', 'SegRescale', 'PhotoMetricDistortion',
    'RandomRotate', 'AdjustGamma', 'CLAHE', 'Rerange', 'RGB2Gray',
    'RandomCutOut', 'RandomMosaic', 'PackSegInputs', 'ResizeToMultiple',
    'LoadImageFromNDArray', 'LoadBiomedicalImageFromFile',
    'LoadBiomedicalAnnotation', 'LoadBiomedicalData', 'GenerateEdge',
    'DecathlonDataset', 'LIPDataset', 'ResizeShortestEdge',
    'BioMedicalGaussianNoise', 'BioMedicalGaussianBlur',
    'BioMedicalRandomGamma', 'BioMedical3DPad', 'RandomRotFlip',
    'SynapseDataset', 'REFUGEDataset', 'MapillaryDataset_v1',
    'MapillaryDataset_v2', 'WatermelonDataset'
]
```

5. prepare WatermelonDataset, uniform file suffix
```python
import os
import shutil
import cv2
from tqdm import tqdm

data_dir = 'Watermelon87_Semantic_Seg_Mask'

for folder_name in tqdm(['img_dir//train', 'img_dir//val'], total=2):
    folder_path = os.path.join(data_dir, folder_name)
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        base_name, extension = os.path.splitext(file_path)
        new_filepath = base_name + '.jpg'
        cv2.imwrite(new_filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        os.remove(file_path)

train_names = os.listdir(os.path.join(data_dir, 'img_dir//train'))
val_names = os.listdir(os.path.join(data_dir, 'img_dir//val'))

images_dir = os.path.join(data_dir, 'images'); os.makedirs(images_dir, exist_ok=True) # images dir
labels_dir = os.path.join(data_dir, 'labels'); os.makedirs(labels_dir, exist_ok=True) # masks dir
splits_dir = os.path.join(data_dir, 'splits'); os.makedirs(splits_dir, exist_ok=True) # train/val file dir

for folder_name in tqdm(['img_dir//train', 'img_dir//val', 'ann_dir//train', 'ann_dir//val'], total=4):
    source = os.path.join(data_dir, folder_name) # copy source
    if 'img_dir' in folder_name:
        destination = images_dir # destination
    elif 'ann_dir' in folder_name:
        destination = labels_dir # destination
    file_names = [name for name in os.listdir(source) if any(name.endswith(file_type) for file_type in ['jpg', 'png'])]
    for name in file_names:
        shutil.copy2(os.path.join(source, name), os.path.join(destination, name)) # copy files to the common folder

with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
    f.writelines(os.path.splitext(name)[0] + '\n' for name in train_names) # creat a txt file for train
with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:
    f.writelines(os.path.splitext(name)[0] + '\n' for name in val_names) # create a txt file for validation
```

6. copy files to the mmsegmentation folder
```python
import os
import shutil
import cv2
from tqdm import tqdm

data_dir = 'Watermelon87_Semantic_Seg_Mask'

list2copy = [
    'Configs//mmseg_watermelon.py',
    'Configs//pspnet_r101-d8_4xb4-80k_ade20k-512x512.py',
    'Configs//pspnet_r101-d8_512x512_80k_ade20k_20200614_031423-b6e782f0.pth'
]

for dir in tqdm(list2copy, total=len(list2copy)):
    shutil.copy2(dir, 'mmsegmentation')

list2copy2 = [
    'Configs//__init__.py',
    'Configs//watermelonDataset.py'
]

for dir in tqdm(list2copy2, total=len(list2copy2)):
    shutil.copy2(dir, 'mmsegmentation//mmseg//datasets')

shutil.copytree(data_dir, 'mmsegmentation//Watermelon87_Semantic_Seg_Mask')
```

7. train PSPNet
```python
%cd mmsegmentation
!mim train mmseg mmseg_watermelon.py
```

## Results

Test on val set
```python
!mim test mmseg ..//Models//PSPNet//mmseg_watermelon.py --checkpoint \
        ..//Models//PSPNet//best_mIoU_epoch_1963.pth --gpus 1
```
```shell
Loads checkpoint by local backend from path: ..//Models//PSPNet//best_mIoU_epoch_1963.pth
06/15 20:54:40 - mmengine - INFO - Load checkpoint from ..//Models//PSPNet//best_mIoU_epoch_1963.pth
06/15 20:54:47 - mmengine - INFO - per class results:
06/15 20:54:47 - mmengine - INFO - 
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background | 98.63 |  99.3 |
|    red     | 97.85 | 98.84 |
|   green    |  94.4 | 96.92 |
|   white    | 92.93 | 96.96 |
| seed-black | 76.98 | 85.63 |
| seed-white | 78.28 |  88.0 |
+------------+-------+-------+
06/15 20:54:47 - mmengine - INFO - Epoch(test) [24/24]    aAcc: 98.7200  mIoU: 89.8500  mAcc: 94.2800  data_time: 0.0067  time: 0.3116
Testing finished successfully.
```

## Inference on test image
```python
%cd ..

from mmseg.utils import register_all_modules
from mmseg.apis import init_model, inference_model, show_result_pyplot

register_all_modules(init_default_scope=False) 
model = init_model('Models//PSPNet//mmseg_watermelon.py', 'Models//PSPNet//best_mIoU_epoch_1963.pth', device='cuda') 
result = inference_model(model, 'Test//IMG_9858.jpg')
vis_iamge = show_result_pyplot(model, 'Test//IMG_9858.jpg', result, out_file='Test//IMG_9858_result.jpg')
```

![IMG_9858.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment4/Test/IMG_9858.jpg))

![IMG_9858_result.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment4/Test/IMG_9858_result.jpg)
