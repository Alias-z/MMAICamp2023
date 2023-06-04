# Using mmpose and mmdet to detect key points on human ears

## Dataset
Created by 张子豪、田文博. Download link: https://pan.baidu.com/s/1swTLpArj7XEDXW4d0lo7Mg, extration code: 741p

## Related tutorials on RTMPose
1. Installation: https://www.bilibili.com/video/BV1Pa4y1g7N7/?spm_id_from=333.788&vd_source=2af21dbdecafe5884922ea5c29d9991c
2. Train mmdet: https://www.bilibili.com/video/BV1Lm4y1879K/?spm_id_from=333.788&vd_source=2af21dbdecafe5884922ea5c29d9991c
3. Train mmpose: https://www.bilibili.com/video/BV12a4y1u7sd/?spm_id_from=333.788&vd_source=2af21dbdecafe5884922ea5c29d9991c

## Tasks
1. Install mmdet and mmpose
```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc3'
mim install "mmdet>=3.0.0rc6"
mim install "mmpose>=1.0.0"
```

2. train RTMdet to detect human ears
```shell
mim train mmdet Configs//rtmdet_ear.py
```

3. train RTMpose to subsequently detect key points on ears
```shell
mim train mmpose Configs//rtmpose_ear.py
```


## Results

1. mmdet
```shell
mim test mmdet Configs//rtmdet_ear.py --checkpoint \
       Models//RTMdet//best_coco_bbox_mAP_epoch_96.pth --gpus 1 
```
```shell
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.18s).
Accumulating evaluation results...
DONE (t=0.02s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.849
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.967
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.967
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.849
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.879
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.879
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.879
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.879
06/04 18:14:00 - mmengine - INFO - bbox_mAP_copypaste: 0.849 0.967 0.967 -1.000 -1.000 0.849
06/04 18:14:00 - mmengine - INFO - Epoch(test) [2/2]    coco/bbox_mAP: 0.8490  coco/bbox_mAP_50: 0.9670  coco/bbox_mAP_75: 0.9670  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.8490  data_time: 4.3082  time: 4.7313
Testing finished successfully.
```

2.mmpose

Note: large learnig rate caused all COCO metrics to be 0.
```shell
mim test mmpose Configs//rtmpose_ear.py --checkpoint \
       Models//RTMPose//best_PCK_epoch_175.pth --gpus 1 
```

```shell
Running per image evaluation...
Evaluate annotation type *keypoints*
DONE (t=0.01s).
Accumulating evaluation results...
DONE (t=0.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.719
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.897
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.755
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.929
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.755
06/04 18:42:53 - mmengine - INFO - Evaluating PCKAccuracy (normalized by ``"bbox_size"``)...
06/04 18:42:53 - mmengine - INFO - Evaluating AUC...
06/04 18:42:53 - mmengine - INFO - Evaluating NME...
06/04 18:42:53 - mmengine - INFO - Epoch(test) [2/2]    coco/AP: 0.718564  coco/AP .5: 1.000000  coco/AP .75: 0.897039  coco/AP (M): -1.000000  coco/AP (L): 0.718564  coco/AR: 0.754762  coco/AR .5: 1.000000  coco/AR .75: 0.928571  coco/AR (M): -1.000000  coco/AR (L): 0.754762  PCK: 0.968254  AUC: 0.112188  NME: 0.043449  data_time: 4.577252  time: 4.835264
Testing finished successfully.
```
