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
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.847
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.967
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.967
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.847
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.876
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.876
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.876
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.876
06/04 16:40:29 - mmengine - INFO - bbox_mAP_copypaste: 0.847 0.967 0.967 -1.000 -1.000 0.847
06/04 16:40:29 - mmengine - INFO - Epoch(test) [3/3]    coco/bbox_mAP: 0.8470  coco/bbox_mAP_50: 0.9670  coco/bbox_mAP_75: 0.9670  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.8470  data_time: 5.2023  time: 5.8892
Testing finished successfully.
```

2.mmpose

Note: large learnig rate caused all COCO metrics to be 0.
```shell

```

updating...
