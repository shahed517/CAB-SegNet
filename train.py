import numpy as np
from loss import TverskyLoss, BoundaryLoss
import torch, os, math, argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from dataset import MonuSeg
from model import Final_Model
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    check_accuracy
)

# Argument Parser
parser = argparse.ArgumentParser(description='NAtt-UNet')
parser.add_argument('--train_dir', type=str, default='MonuSeg dataset/train_folder')
parser.add_argument('--val_dir', type=str, default='MonuSeg dataset/val_folder')
parser.add_argument('--ckpt_dir', type=str, default='Home/ckpt')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=400)

parser.add_argument('--img_size', type=int, default=512, help='training image size')
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.0005)

parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--pin_memory', type=bool, default=True)
parser.add_argument('--reduceLR_patience', type=int, default=50)
parser.add_argument('--reduceLR_factor', type=float, default=0.5)
parser.add_argument('--loadModel_dir', type=str, default='Home/weight.pth.tar')
args = parser.parse_args()


LEARNING_RATE = args.lr
BATCH_SIZE = args.batchsize
ALPHA = args.alpha
BETA = args.beta
WEIGHT_DECAY = args.weight_decay
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = args.epochs
NUM_WORKERS = args.num_workers

IMAGE_HEIGHT = args.img_size
IMAGE_WIDTH = args.img_size
PIN_MEMORY = args.pin_memory

TRAIN_IMG_DIR = os.path.join(args.train_dir, 'img/')
TRAIN_MASK_DIR = os.path.join(args.train_dir, 'labelcol/')
VAL_IMG_DIR = os.path.join(args.val_dir, 'img/')
VAL_MASK_DIR = os.path.join(args.val_dir, 'labelcol/')

LOAD_MODEL = False
LOAD_MODEL_DIR = args.loadModel_dir

val_f1 = []; val_iou = [];


def train_fn_AMP(loader, model, optimizer, loss1, loss2, loss3, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE) # for binary ce loss
        # unsqueeze adds a channel dimension         
        optimizer.zero_grad() 
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            l1 = loss1(predictions[0], targets) # mask
            l2 = loss2(predictions[1], targets) # boundary
            l3 = loss3(torch.sigmoid(predictions[0]), targets) # [1] is the final output
            loss = ALPHA*l1 + (1-ALPHA)*l2  + BETA*l3
        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.RandomCrop(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.1),            
            A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=0.1),
            A.Cutout(num_holes=4, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.1),
            A.Blur(blur_limit=7, always_apply=False, p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()    
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()    
        ],
    )

    model = Final_Model()
    model = model.to(DEVICE)

    loss1 = TverskyLoss().to(DEVICE)
    loss2 = BoundaryLoss().to(DEVICE)
    loss3 = nn.MSELoss().to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS
    )
  
    if LOAD_MODEL:
        checkpoint = torch.load(LOAD_MODEL_DIR)
        load_checkpoint(checkpoint, model)
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    scaler = torch.cuda.amp.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = args.reduceLR_factor, patience = args.reduceLR_patience, verbose = True) 

    print('Initiating/Resuming Training...')

    best_val_score = 0; best_f1_score = 0; 
    for epoch in range(NUM_EPOCHS):
        # train_fn handles only 1 epoch
        train_fn_AMP(train_loader, model, optimizer, loss1, loss2, loss3, scaler)
     
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        
        # check scores on validation set
        iou_score_val, f1_score_val, _, _, _, _ = check_accuracy(val_loader, model, device=DEVICE)
        print(f"EPOCH COMPLETED : {epoch + 1}")
        print(f"Mean IoU score on validation set : {iou_score_val}")
        print(f"Mean F1 score on validation set : {f1_score_val}")

        if iou_score_val > best_val_score and f1_score_val > best_f1_score:
          save_checkpoint(args.ckpt_dir, checkpoint, epoch + 1)
          best_val_score = iou_score_val
          best_f1_score = f1_score_val
        val_f1.append(f1_score_val)
        val_iou.append(iou_score_val)

        scheduler.step(iou_score_val + f1_score_val)

if __name__ == '__main__':
    main()





