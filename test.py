import numpy as np
import torch, os, math, argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from dataset import MoNuSeg
from model import Final_Model
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    check_accuracy,
    produce_output_masks
)

# Argument Parser
parser = argparse.ArgumentParser(description='RA-SegNet')
parser.add_argument('--test_dir', type=str, required = True, default='MonuSeg dataset/test_folder')
parser.add_argument('--weights_dir', type=str, required = True, default='xyz/weights.pth.tar')
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--pin_memory', type=bool, default=True)
parser.add_argument('--img_savedir', type=bool, required = True, default='abc/img_folder')
args = parser.parse_args()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = args.num_workers

IMAGE_HEIGHT = args.img_size
IMAGE_WIDTH = args.img_size
PIN_MEMORY = args.pin_memory

TEST_IMG_DIR = os.path.join(args.test_dir, 'img/')
TEST_MASK_DIR = os.path.join(args.test_dir, 'labelcol/')
path = args.img_savedir
LOAD_MODEL_DIR = args.weights_dir


def main():

    model = Final_Model()
    model = model.to(DEVICE)

    
    test_transforms = A.Compose(
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
    test_ds = MoNuSeg(
        image_dir = TEST_IMG_DIR, 
        mask_dir = TEST_MASK_DIR,   
        transform=test_transforms
    )
      
    test_loader = DataLoader(
        test_ds,
        batch_size = 1,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,            
    )  
  
    checkpoint = torch.load(LOAD_MODEL_DIR)
    load_checkpoint(checkpoint, model)

    iou_score_test, f1_score_test, precision, recall, accuracy, dice_score = check_accuracy(test_loader, model, device=DEVICE)
 
    print(f"Mean IoU score on test set : {iou_score_test}")
    print(f"Mean F1 score on test set : {f1_score_test}")
    print(f"Mean Precision score on test set in : {precision}")
    print(f"Mean Recall score on test set in : {recall}")
    print(f"Mean Accuracy score on test set in : {accuracy}")
    print(f"Mean Dice score on test set in : {dice_score}")

    produce_output_masks(test_loader, model, path)

if __name__ == '__main__':
    main()





