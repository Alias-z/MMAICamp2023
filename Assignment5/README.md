# Using mmagic to design a room from scratch

## Dataset
Will be generated using mmagic

## Related tutorials on mmagic
1. mmediting introduction: https://www.bilibili.com/video/BV1hu4y1o7jU/?spm_id_from=333.999.0.0
2. mmagic code demo: https://www.bilibili.com/video/BV1gM4y1n7vP/?spm_id_from=333.999.0.0&vd_source=2af21dbdecafe5884922ea5c29d9991c


## Tasks
1. Install mmagic
```python
!pip install clip transformers gradio 'httpx[socks]' diffusers==0.14.0
!pip install -U openmim
!mim install mmengine
!mim install 'mmcv>=2.0.0rc3'
!mim install 'mmdet>=3.0.0'
!rm -rf mmagic
!git clone https://github.com/open-mmlab/mmagic.git
%cd mmagic
!pip install -e .
%cd ..
```

2. generate our unfurnished room from text prompt 'an unfurnished room'
```python
from mmagic.apis import MMagicInferencer

sd_inferencer = MMagicInferencer(model_name='stable_diffusion')
text_prompts = 'an unfurnished room'

sd_inferencer.infer(text=text_prompts, result_out_dir='Results//test2image.png')
```
![test2image.png](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment5/Results/test2image.png)


3. generate a furnished room pics to pics with ControlNet 
<br> we have to detect the edges on the unfurnished image first
<br> then describe our dream room with prompt 'A fully furnished room with colorful wallpapers'
```python
import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

cfg = Config.fromfile('mmagic//configs//controlnet//controlnet-canny.py')
controlnet = MODELS.build(cfg.model).cuda()

control_img = mmcv.imread('Results//test2image.png')
control = cv2.Canny(control_img, 100, 200)
control = control[:, :, None]
control = np.concatenate([control] * 3, axis=2)
control = Image.fromarray(control)

prompt = 'A fully furnished room with colorful wallpapers'

output_dict = controlnet.infer(prompt, control=control)
samples = output_dict['samples']
for idx, sample in enumerate(samples):
    sample.save(f'Results//sample_{idx}.png')
controls = output_dict['controls']
for idx, control in enumerate(controls):
    control.save(f'Results//control_{idx}.png')
```
![control_0.png](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment5/Results/control_0.png)
![sample_0.png](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment5/Results/sample_0.png)

4. finally, we could show our dear customer a poster with our design
```python
def imread_rgb(image_dir):
    """cv2.imread + BRG2RGB """
    image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
    return image

unfurnished = imread_rgb('Results//test2image.png')
edge = imread_rgb('Results//control_0.png')
furnished = imread_rgb('Results//sample_0.png')

# concatenate images horizontally (along the column axis)
image = cv2.hconcat([unfurnished, edge, furnished])

# create a black image with the same width as the concatenated image for writing text
text_image = np.zeros((100, image.shape[1], 3), np.uint8)

# define font, scale, color, thickness, and other properties.
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2; font_color = (255, 255, 255); line_type = 2
text = 'here is your furnished room!'
(text_width, text_height) = cv2.getTextSize(text, font, font_scale, line_type)[0]
text_x = (text_image.shape[1] - text_width) // 2
text_y = (text_image.shape[0] + text_height) // 2


cv2.putText(text_image, 
            text, (text_x, text_y),
            font,  font_scale, font_color, line_type)

poster = cv2.vconcat([image, text_image])
cv2.imwrite('Results//poster.jpg', cv2.cvtColor(poster, cv2.COLOR_RGB2BGR))
```
![poster.jpg](https://github.com/Alias-z/mmcamp2023/blob/main/Assignment5/Results/poster.jpg)

