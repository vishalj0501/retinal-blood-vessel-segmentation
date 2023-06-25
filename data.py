import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

class RetinaDataset(Dataset):
    def __init__(self, images_path, masks_path):

        self.images_path = images_path
        # print("---1--")
        # print(self.images_path)
        # print("---1--")
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        # print("---2--")
        # print(self.n_samples)
        # print("---2--")

    # def __getitem__(self, index):
    #     image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
    #     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image = image/255.0 ## (512, 512, 3)
    #     image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
    #     #image = torch.permute(image, (2, 0, 1))  ## (3, 512, 512)
    #     image = image.astype(np.float32)
    #     image = torch.from_numpy(image)

    #     mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
    #     mask = mask/255.0   ## (512, 512)
    #     mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
    #     mask = mask.astype(np.float32)
    #     mask = torch.from_numpy(mask)

    #     return image, mask

    def __getitem__(self, index):
        image_path = self.images_path[index]
        # image_path = os.path.join(self.images_path, self.images_path[index])

        # print("---")
        # print(image_path)
        # print("---")
        # mask_path = os.path.join(self.masks_path, self.masks_path[index])
        mask_path = self.masks_path[index]

        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load image at path: {image_path}")
            image = image / 255.0  # Normalize image between 0 and 1
            image = np.transpose(image, (2, 0, 1))  # Transpose dimensions to (3, 512, 512)
            image = torch.tensor(image, dtype=torch.float32)
        except Exception as e:
            # Handle any exceptions that occur during image loading
            print(f"Error loading image: {str(e)}")
            raise e

        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask at path: {mask_path}")
            mask = mask / 255.0  # Normalize mask between 0 and 1
            mask = np.expand_dims(mask, axis=0)  # Add channel dimension (1, 512, 512)
            mask = torch.tensor(mask, dtype=torch.float32)
        except Exception as e:
            # Handle any exceptions that occur during mask loading
            print(f"Error loading mask: {str(e)}")
            raise e

        return image, mask


    def __len__(self):
        return self.n_samples