# CAB-SegNet
This Repository contains the Pytorch implementation of our study:

"**CAB-SegNet: A Context Aware Boundary Preserving Dual-Stage Network for Accurate Nucleus Segmentation**"


<!-- ![main block diagram latest](https://user-images.githubusercontent.com/85427219/141252295-a55728f8-7255-4746-ab08-820c5e73bb73.jpg) -->

![main block diagram FINAL](https://user-images.githubusercontent.com/85427219/152217009-e7333fb1-9d62-477e-80cf-945a5e6f957b.jpg)

<!-- ![attention_block_diagram](https://user-images.githubusercontent.com/85427219/141128709-943f6ad7-6491-4e53-9c91-3177328ff7a8.jpg) -->

## Datasets used in the study:
1. MoNuSeg ([Link to Original](https://monuseg.grand-challenge.org/Data/), [Link to Processed Data](https://drive.google.com/drive/folders/1eUVucH9qhhyVsq22UtU1VXfdSe-yDco7?usp=sharing))
2. TNBC ([Link to Original](https://zenodo.org/record/1175282#.YW73qRpByUl))
3. Data Science Bowl 2018 dataset ([Link to Original](https://www.kaggle.com/c/data-science-bowl-2018))

## Usage
### Clone this repository
```
git clone https://github.com/shahed517/CAB-SegNet
cd CAB-SegNet
```
### Python requirements
The code is built upon Python 3.7

- numpy
- PyTorch (>=1.7.0)
- torchvision
- scipy
- scikit-image
- tqdm
- opencv
- albumentations
- Pillow

<!-- ### Download pretrained weights
Download the pretrained weights from the [Google Drive Folder](https://drive.google.com/drive/folders/1wtsQrl5vgl9SKfexMSfhb8QAPnCoUZas?usp=sharing) -->

### Preparing custom datasets
The custom dataset should be prepared in the following format; however the image names need not necessarily be '0001', '0002' etc. Keeping the names same for the image-mask pairs should be enough. 
```
train folder-----
      img----
          0001.png
          0002.png
          .......
      labelcol---
          0001.png
          0002.png
          .......
validation folder-----
      img----
          0001.png
          0002.png
          .......
      labelcol---
          0001.png
          0002.png
          .......
```

### Training example
```
python train.py --train_dir "write your train folder here" \
                --val_dir "write your validation folder here" \
                --ckpt_dir "write your checkpoint save directory here" \
                --epochs 100 \
                --lr 0.001 \
                --img_size 256 \
                --batchsize 4
```                
### Testing example
```
python test.py --test_dir "write your test folder here" --img_size 256 --weights_dir "write your weightpath directory here" --img_savedir "save output images here"
``` 

<!-- ## Citation
Please cite the following paper if you find the code useful in your work: -->
