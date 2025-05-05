# Visual Class Incremental Learning with Textual Priors Guidance based on an Adapted Vision-Language Model
This is the official repository for Pytorch Implementation of "Visual Class Incremental Learning with Textual Priors Guidance based on an Adapted Vision-Language Model". 




## Installation
```bash
git clone https://github.com/Fatflower/CIL_Adapted_VLM.git
cd CIL_Adapted_VLM
pip install -r requirements.txt


```


## Data Preparation

Before running the project, make sure you have the necessary datasets ready. All input images are resized or cropped to **224 × 224** pixels.




## Training

```bash
CUDA_VISIBLE_DEVICES=0 python main_test.py --config options/clip_two/imagenet_r.yaml

```



## Test

```bash

CUDA_VISIBLE_DEVICES=0 python main_test.py --config options/clip_two/imagenet_r.yaml --test_dir logs/multi_step/clip_two_cil_replay_None/imagenetr_i2t/clip_vit_b_16_224_b20i20/seed_100

```


