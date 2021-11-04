# RA-SegNet
This Repository contains the Pytorch implementation of our paper :

"**RA-SegNet: An Attention Guided Dual-Stage Encoder-Decoder Framework with Boundary Constraint for Accurate Nucleus Segmentation**"

![main_block_diagram](https://user-images.githubusercontent.com/85427219/140326989-6fcd999b-d324-408b-b975-f803813d0d81.jpg)

## Datasets used in the paper:
1. MoNuSeg ([Link to Original](https://monuseg.grand-challenge.org/Data/), [Link to Processed Data](https://drive.google.com/drive/folders/1eUVucH9qhhyVsq22UtU1VXfdSe-yDco7?usp=sharing))
2. TNBC ([Link to Original](https://zenodo.org/record/1175282#.YW73qRpByUl))
3. Data Science Bowl 2018 dataset ([Link to Original](https://www.kaggle.com/c/data-science-bowl-2018))

## Preparing custom datasets
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

## Training example
```
python train.py --train_dir "write your train folder here" \
                --val_dir "write your validation folder here" \
                --ckpt_dir "write your checkpoint save directory here" \
                --epochs 100 \
                --lr 0.001 \
                --img_size 256 \
                --batchsize 4
```                
## Testing example
```
python test.py --test_dir "write your test folder here" --img_size 256 --weights_dir "write your weightpath directory here" --img_savedir "save output images here"
``` 

<!-- ## Citation
Please cite the following paper if you find the code useful in your project/work: -->
