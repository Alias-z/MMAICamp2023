# Using mmdet to detect drinks

## Dataset
Created by 张子豪 田文博. Download link: https://zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Drink_284_Detection_Dataset/Drink_284_Detection_coco.zip 
<br> (Alipay 5 RMB to 407431120@qq.com for maintainance)

## Related tutorials on mmdet
1. mmdet introduction: https://www.bilibili.com/video/BV1Ak4y1p7W9/?spm_id_from=333.999.0.0
2. mmdet code demo: https://www.bilibili.com/video/BV1Tm4y1q7fy/?spm_id_from=333.999.0.0&vd_source=2af21dbdecafe5884922ea5c29d9991c


## Tasks
1. Install mmdet
```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc3'
mim install "mmdet>=3.0.0rc6"
```

2. config RTMDet-tiny
```python
!mim search mmdet --model RTMdet
!mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest Configs
```

3. modify config file accordingly and train ResNet-50
```shell
mim train mmdet Configs//rtmdet_drinks.py
```

4. Visualize feature map and grad CAM (shown below)

## Results

Test on val set
```shell
mim test mmdet Configs//rtmdet_drinks.py --checkpoint \
       Models//RTMdet//best_coco_bbox_mAP_epoch_100.pth --gpus 1 
```
```shell
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.69s).
Accumulating evaluation results...
DONE (t=0.16s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.965
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.994
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.994
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.965
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.947
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.976
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.976
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.976
06/10 20:56:33 - mmengine - INFO - bbox_mAP_copypaste: 0.965 0.994 0.994 -1.000 -1.000 0.965
06/10 20:56:33 - mmengine - INFO - Epoch(test) [2/2]    coco/bbox_mAP: 0.9650  coco/bbox_mAP_50: 0.9940  coco/bbox_mAP_75: 0.9940  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.9650  data_time: 8.5729  time: 10.0379
Testing finished successfully.
```

## Inference on test image
```python
import cv2
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector
from mmdet.apis import inference_detector

register_all_modules(init_default_scope=False)
model = init_detector('Configs//rtmdet_drinks.py', 'Models//RTMdet//best_coco_bbox_mAP_epoch_100.pth', device='cuda')

images_dir = os.listdir('Test')
images_dir = [os.path.join('Test', image) for image in images_dir]
images = [cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_RGB2BGR) for image_dir in images_dir]
predictions = [inference_detector(model, image) for image in images]

from matplotlib import pyplot as plt
from mmdet.registry import VISUALIZERS

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

os.makedirs('Results', exist_ok=True)
for idx, image in enumerate(images):
    visualizer.add_datasample('result', image, data_sample=predictions[idx], pred_score_thr=0.5)
    result = visualizer.get_image()
    plt.imshow(result); plt.axis('off')
    plt.savefig('Results//result_n.jpg'.replace('n', str(idx)))
    plt.show()
```

![result_0.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment3/Results/result_0.jpg))

![result_1.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment3/Results/result_1.jpg)

## Feature map visualization

1. install mmyolo
```shell
!rm -rf mmyolo
!git clone -b tutorials https://github.com/open-mmlab/mmyolo.git
%cd mmyolo
!pip install -e .
%cd ..
```

2. check backbone

```python
from PIL import Image

!python mmyolo/demo/featmap_vis_demo.py \
      Test/IMG_9850.jpg \
      Configs//rtmdet_drinks.py \
      Models//RTMdet//best_coco_bbox_mAP_epoch_100.pth  \
      --preview-model\
      --target-layers backbone  \
      --channel-reduction squeeze_mean \
      --out-dir Feature_map/backbone
plt.imshow(cv2.cvtColor(cv2.imread('Feature_map//backbone//IMG_9850.jpg'), cv2.COLOR_RGB2BGR))
plt.axis('off'); plt.title('backbone feature map')
```

![backbone%20IMG_9850.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment3/Feature_map/backbone%20IMG_9850.jpg)

3. check neck

```python
!python mmyolo/demo/featmap_vis_demo.py \
      Test/IMG_9850.jpg \
      Configs//rtmdet_drinks.py \
      Models//RTMdet//best_coco_bbox_mAP_epoch_100.pth  \
      --target-layers neck  \
      --channel-reduction squeeze_mean \
      --out-dir Feature_map/neck
plt.imshow(cv2.cvtColor(cv2.imread('Feature_map//neck//IMG_9850.jpg'), cv2.COLOR_RGB2BGR))
plt.axis('off'); plt.title('neck feature map')
```
![neck%20IMG_9850.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment3/Feature_map/neck%20IMG_9850.jpg)

## Grad box AM visualization

1. install grad-cam
```shell
pip install grad-cam
```

2. check backbone.stage4
```python
!python mmyolo/demo/boxam_vis_demo.py \
      Test/IMG_9850.jpg \
      Configs//rtmdet_drinks.py \
      Models//RTMdet//best_coco_bbox_mAP_epoch_100.pth  \
      --target-layers backbone.stage4  \
      --out-dir Grad_CAM/backbone
plt.imshow(cv2.cvtColor(cv2.imread('Grad_CAM//backbone//IMG_9850.jpg'), cv2.COLOR_RGB2BGR))
plt.axis('off'); plt.title('backbone.stage4 grad CAM map')
```
![backbone%20IMG_9850.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment3/Grad_CAM/backbone%20IMG_9850.jpg)

3. check neck.out_convs[2](min)
```python
!python mmyolo/demo/boxam_vis_demo.py \
      Test/IMG_9850.jpg \
      Configs//rtmdet_drinks.py \
      Models//RTMdet//best_coco_bbox_mAP_epoch_100.pth  \
      --target-layers neck.out_convs[2]  \
      --out-dir Grad_CAM/neck_min
plt.imshow(cv2.cvtColor(cv2.imread('Grad_CAM//neck_min//IMG_9850.jpg'), cv2.COLOR_RGB2BGR))
plt.axis('off'); plt.title('neck.out_convs[2](min) grad CAM map')
```
![neck_min%20IMG_9850.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment3/Grad_CAM/neck_min%20IMG_9850.jpg)

4. check neck.out_convs[0](max)
```python
!python mmyolo/demo/boxam_vis_demo.py \
      Test/IMG_9850.jpg \
      Configs//rtmdet_drinks.py \
      Models//RTMdet//best_coco_bbox_mAP_epoch_100.pth  \
      --target-layers neck.out_convs[0]  \
      --out-dir Grad_CAM/neck_max
plt.imshow(cv2.cvtColor(cv2.imread('Grad_CAM//neck_max//IMG_9850.jpg'), cv2.COLOR_RGB2BGR))
plt.axis('off'); plt.title('neck.out_convs[0](max) grad CAM map')
```
![neck_max%20IMG_9850.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment3/Grad_CAM/neck_max%20IMG_9850.jpg)
