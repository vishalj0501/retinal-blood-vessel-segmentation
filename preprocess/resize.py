import cv2
import os
from tqdm import tqdm
import imageio
import glob as glob


train_augmented_path_images = 'data/augmented/training/images'
train_augmented_path_1st_manual = 'data/augmented/training/1st_manual'

test_augmented_path_images = 'data/augmented/test/images'
test_augmented_path_1st_manual = 'data/augmented/test/1st_manual'

def resize_images_masks(images, masks, save_path_images, save_path_masks, size=512):
    for (image, mask) in tqdm(zip(images, masks), total=len(images)):
        name = image.split('/')[-1].split('.')[0]
        image = imageio.imread(image)
        mask = imageio.imread(mask)
        image = cv2.resize(image, (size, size))
        mask = cv2.resize(mask, (size, size))
        imageio.imwrite(os.path.join(save_path_images, f'{name}.png'), image)
        imageio.imwrite(os.path.join(save_path_masks, f'{name}_mask.png'), mask)


train_images = sorted(glob.glob(f'{train_augmented_path_images}/*'))
train_masks = sorted(glob.glob(f'{train_augmented_path_1st_manual}/*'))
test_images = sorted(glob.glob(f'{test_augmented_path_images}/*'))
test_masks = sorted(glob.glob(f'{test_augmented_path_1st_manual}/*'))

resize_images_masks(train_images, train_masks, train_augmented_path_images, train_augmented_path_1st_manual, size=512)
resize_images_masks(test_images, test_masks, test_augmented_path_images, test_augmented_path_1st_manual, size=512)
