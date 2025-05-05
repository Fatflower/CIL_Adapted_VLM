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

##  Prepare Pretrained Models

1. Create a directory:
    ```bash
    mkdir -p pretrain_weights/clip_vit_base_patch16
    ```

2. Download the pretrained model from:  
   [https://huggingface.co/openai/clip-vit-base-patch16/tree/main](https://huggingface.co/openai/clip-vit-base-patch16/tree/main)

3. Place the downloaded files into the `pretrain_weights/clip_vit_base_patch16/` directory.

---


## Training

```bash
CUDA_VISIBLE_DEVICES=0 python main_test.py --config options/clip_two/imagenet_r.yaml

```
**Note:**  
 You can replace `imagenet_r.yaml` with any dataset-specific configuration file, such as:

 - `options/clip_two/cifar100.yaml`
 - `options/clip_two/skin40.yaml`
 - `options/clip_two/your_dataset.yaml`


## Test

```bash

CUDA_VISIBLE_DEVICES=0 python main_test.py --config options/clip_two/imagenet_r.yaml --test_dir logs/multi_step/clip_two_cil_replay_None/imagenetr_i2t/clip_vit_b_16_224_b20i20/seed_100

```

Make sure that:

- `--config` points to the correct dataset configuration.
- `--test_dir` points to the directory containing your trained model logs and checkpoints.


