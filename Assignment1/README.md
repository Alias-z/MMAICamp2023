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

2. train RTMpose to subsequently detect key points on ears
```shell
mim train mmpose Configs//rtmpose_ear.py
```
