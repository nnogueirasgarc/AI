import torch
import torchvision.transforms as Transforms
from torch.utils.data import Dataset
import numpy as np
import random

from library import load_AIRR_images 

# Utility functions for transforming images and masks from/to PIL images and tensors:
transform_tensor = Transforms.ToTensor()
transform_PIL = Transforms.ToPILImage()

class AIRR_Dataset(Dataset):
    """ 
    Dataset class for the AIRR dataset
    """

    def __init__(self):

        print('Load images')
        [masks, images] = load_AIRR_images()
        self.X = images
        self.t = masks

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        """
        TODO:
        Part 3: 
            - Modify this function so as to implement a horizontal flip
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Images:
        im = self.X[idx]
        im = transform_tensor(im)  # Convert NumPy array to tensor

        # Masks:
        t = torch.tensor(self.t[idx])
        t = torch.reshape(t, (1, t.shape[0], t.shape[1]))
        t = t.float() / 255.0
        
        if torch.rand(1) < 0.5:
            im = torch.flip(im, dims=[2])
            t = torch.flip(t, dims=[-1])

        sample = im, t
        return sample