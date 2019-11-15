import math
import os
import pickle
import sys
# pytorch
import torch
from torch.nn.modules.loss import _Loss
import torchvision.transforms as transforms
import torch.utils.data as torch_data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# models
from models.cdnet import CDNet
from models.dncnn import DnCNN
# stats
import matplotlib.pyplot as plt
import numpy as np
# scale
import cv_practical.main as cvp_utils
# fancy stuff
from tqdm import tqdm


class CIRCLEDataset(torch_data.Dataset):
    """Dataset for training CDNet.

    This dataset inherits from PyTorch torch_data.Dataset
    to be used to create batches for PyTorch's dataloader.
    Each sample in the dataset is a pair of noisy image (I_n) and
    the parameters of the circle in that image (x, y, r).
    """

    def __init__(self, count=1000, noise = 1,
                 random_noise = True, debug = False):
        """
        Args:
            count: Number of sample image pairs in the dataset.
            noise: Max level of gaussian additive noise.
            random_noise: If set to True, additive gausian noise
                is multiplied by a random constant sampled from a
                uniform distruibution in range (0, noise)
            debug: If set to True, writes image pairs to disk.
        """

        self._deubg = debug
        self._count = count
        self._noise = noise
        self._random_noise = random_noise
        self._circle_images, self._circle_params = [], []

        self.normalize = lambda a: (a-np.min(a))/(np.max(a)-np.min(a))

        self.create_dataset()

    def create_dataset(self):
        for i in range(self._count):
            noise = np.random.uniform(0, self._noise) if \
                self._random_noise else self._noise

            params, img, img_noise = cvp_utils.noisy_circle(200, 50, noise)
            # normalize
            img = self.normalize(img)
            # add a channel axis to conform to
            # PyTorch's tensor specifications.
            self._circle_images.append(
                np.expand_dims(img, axis=0)
            )

            # normalize params.
            self._circle_params.append((np.asarray([
                    (params[0]-100)/100.0,
                    (params[1]-100)/100.0,
                    (params[2]-10)/40.0
                ], dtype = np.float32)
            ))

    def __len__(self):
        return len(self._circle_images)

    def __getitem__(self, idx):
        return [
            self._circle_images[idx], self._circle_params[idx]
        ]



class DnCNNDataset(torch_data.Dataset):
    """Dataset for training our simplified DnCNN.

    This dataset inherits from PyTorch torch_data.Dataset
    to be used to create batches for PyTorch's dataloader.
    Each sample in the dataset is a pair of noisy image (I_n) and
    denoised image (I).
    """

    def __init__(self, count=1000, noise = 1,
                 random_noise = True, debug = False):
        """
        Args:
            count: Number of sample image pairs in the dataset.
            noise: Max level of gaussian additive noise.
            random_noise: If set to True, additive gausian noise
                is multiplied by a random constant sampled from a
                uniform distruibution in range (0, noise)
            debug: If set to True, writes image pairs to disk.
        """

        self._deubg = debug
        self._count = count
        self._noise = noise
        self._random_noise = random_noise
        self._circle_images = []

        self.normalize = lambda a: (a-np.min(a))/(np.max(a)-np.min(a))

        self.create_dataset()

    def create_dataset(self):
        for i in range(self._count):
            noise = np.random.uniform(0, self._noise) \
                if self._random_noise else self._noise

            params, img, img_noise = cvp_utils.noisy_circle(200, 50, noise)
            # normalize.
            img = self.normalize(img)
            img_noise = self.normalize(img_noise)
            # add a channel axis to conform to
            # PyTorch's tensor specifications.
            self._circle_images.append(
                [
                    np.expand_dims(img_noise, axis=0),
                    np.expand_dims(img, axis=0)
                ]
            )

    def __len__(self):
        return len(self._circle_images)

    def __getitem__(self, idx):
        return self._circle_images[idx]
