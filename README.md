# [ECCV2022] EqInv

This repository contains the official PyTorch implementation of paper "Equivariance and Invariance Inductive Bias for Learning from Insufficient Data".

**Equivariance and Invariance Inductive Bias for Learning from Insufficient Data** <br />
[Tan Wang](https://wangt-cn.github.io/), [Qianru Sun](https://qianrusun.com/), Sugiri Pranata, Karlekar Jayashree, [Hanwang Zhang](https://www.ntu.edu.sg/home/hanwangzhang/) <br />
**European Conference on Computer Vision (ECCV), 2022** <br />
**[[Paper: Comming Soon]()] [[Poster: Comming Soon]()] [[Slides: Comming Soon]()]**<br />



## Prerequisites
- Python 3.7
- PyTorch 1.9.0
- tqdm
- randaugment
- opencv-python




## Data Preparation
1. Please download dataset in this [link](https://drive.google.com/drive/u/1/folders/14Ppy9Cfp4gz9AP1tYIGW_XaOjycOIjX_) and put it into the `data` folder.

   

## Training

#### 1. Run the baseline model#1 —— Training From Scratch

   ```
CUDA_VISIBLE_DEVICES=0,1,2,3 python baseline.py -b 256 --name vipriors10_rn50 -j 16 --lr 0.1 data/imagenet_10
   ```

   You can also try the built-in augmentation algorithms, such as `Mixup`

   ```
CUDA_VISIBLE_DEVICES=0,1,2,3 python baseline.py -b 256 --name vipriors10_rn50_mixup -j 16 --lr 0.1 data/imagenet_10 --mixup
   ```

   

#### 2. Run the baseline model#2 —— Training From SSL

   ```
CUDA_VISIBLE_DEVICES=4,5,6,7 python baseline_eq_ipirm.py -b 256 --name vipriors10_rn50_lr0.1_ipirm -j 16 --lr 0.1 data/imagenet_10 --pretrain_path phase1_ssl_methods/run_imagenet10/ipirm_imagenet10/model_ipirm.pth
   ```

For the SSL pretraining process, please follow the following chapter.



#### 3. Run our EqInv model

**Step-1: SSL Pretraining (Equivariance Learning)**

Please follow the original codebase. We list the code we used below:

- MoCo-v2: https://github.com/facebookresearch/moco

- Simsiam: https://github.com/facebookresearch/simsiam

- IP-IRM: https://github.com/Wangt-CN/IP-IRM

- MAE: https://github.com/facebookresearch/mae



**Step-2&3: Downstream Fine-tuning (Invariance Learning)**

   ```
CUDA_VISIBLE_DEVICES=4,5,6,7 python vipriors_eqinv_ready.py -b 128  --name vipriors10_ipirm_mask_sigmoid_rex100._start10 -j 64 data/imagenet_10 --pretrain_path phase1_ssl_methods/run_imagenet10/ipirm_imagenet10/model_ipirm.pth --inv rex --inv_weight 100. --opt_mask --activat_type sigmoid --inv_start 10 --mlp --stage1_model ipirm --num_shot 10
   ```

   You can also adopt `Random Augmentation` to achieve better results

   ```
CUDA_VISIBLE_DEVICES=4,5,6,7 python vipriors_eqinv_ready.py -b 128  --name vipriors10_ipirm_mask_sigmoid_rex10._start10_randaug -j 64 data/imagenet_10 --pretrain_path phase1_ssl_methods/run_imagenet10/ipirm_imagenet10/model_ipirm.pth --inv rex --inv_weight 10. --opt_mask --activat_type sigmoid --inv_start 10 --mlp --stage1_model ipirm --num_shot 10 --random_aug
   ```

   

