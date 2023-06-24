import cv2
import os
import numpy as np
import glob as glob
import imageio
import torch
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90

train_path_1st_manual='data/DRIVE/training/1st_manual'
train_path_images='data/DRIVE/training/images'

test_path_1st_manual='data/DRIVE/test/1st_manual'
test_path_images='data/DRIVE/test/images'

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

train_augmented_path_images = 'data/augmented/training/images'
create_dir(train_augmented_path_images)
train_augmented_path_1st_manual = 'data/augmented/training/1st_manual'
create_dir(train_augmented_path_1st_manual)

test_augmented_path_images = 'data/augmented/test/images'
create_dir(test_augmented_path_images)
test_augmented_path_1st_manual = 'data/augmented/test/1st_manual'
create_dir(test_augmented_path_1st_manual)


def augment_data(images, masks, save_path_images,save_path_masks, augment=True):
    i = 0
    for (image, mask) in tqdm(zip(images, masks), total=len(images)):
        name = image.split('/')[-1].split('.')[0]
        image = imageio.imread(image)
        mask = imageio.imread(mask)
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=image, mask=mask)
            x1 = augmented['image']
            y1 = augmented['mask']
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=image, mask=mask)
            x2 = augmented['image']
            y2 = augmented['mask']
            aug = RandomRotate90(p=1.0)
            augmented = aug(image=image, mask=mask)
            x3 = augmented['image']
            y3 = augmented['mask']
            imageio.imwrite(os.path.join(save_path_images, f'{name}_{i}.png'), image)
            imageio.imwrite(os.path.join(save_path_images, f'{name}_{i+1}.png'), x1)
            imageio.imwrite(os.path.join(save_path_images, f'{name}_{i+2}.png'), x2)
            imageio.imwrite(os.path.join(save_path_images, f'{name}_{i+3}.png'), x3)
            imageio.imwrite(os.path.join(save_path_masks, f'{name}_{i}_mask.png'), mask)
            imageio.imwrite(os.path.join(save_path_masks, f'{name}_{i+1}_mask.png'), y1)
            imageio.imwrite(os.path.join(save_path_masks, f'{name}_{i+2}_mask.png'), y2)
            imageio.imwrite(os.path.join(save_path_masks, f'{name}_{i+3}_mask.png'), y3)
            i += 4
        else:
            imageio.imwrite(os.path.join(save_path_images, f'{name}.png'), image)
            imageio.imwrite(os.path.join(save_path_masks, f'{name}_mask.png'), mask)

train_images = sorted(glob.glob(f'{train_path_images}/*'))
train_masks = sorted(glob.glob(f'{train_path_1st_manual}/*'))
test_images = sorted(glob.glob(f'{test_path_images}/*'))
test_masks = sorted(glob.glob(f'{test_path_1st_manual}/*'))

augment_data(train_images, train_masks, train_augmented_path_images,train_augmented_path_1st_manual, augment=True)
augment_data(test_images, test_masks, test_augmented_path_images,test_augmented_path_1st_manual, augment=False)