import math
import os
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
from models.dncnn import DnCNN
# stats
import matplotlib.pyplot as plt
import numpy as np
# scale
import cv_practical.main as cvp_utils
# fancy stuff
from tqdm import tqdm


class CDNet(nn.Module):
    def __init__(
        self,
        in_planes,
        bbone,
        bbone_weights = None
    ):
        super(CDNet, self).__init__()

        self.denoiser = bbone

        if bbone_weights is not None:
            self._init_bbone(bbone_weights)
        self.conv1 = nn.Conv2d(in_planes, 6, kernel_size = 5, stride = 1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5, stride = 1)
        #self.conv3 = nn.Conv2d(16, 120, kernel_size = 5, stride = 1)

        self.fc1   = nn.Linear(16*47*47, 47*47)
        self.fc2   = nn.Linear(47*47, 3)

        self.name = "cdnet"

    def forward(self, x):
        b_dx, c_dx, w, h = x.size()
        out = self.denoiser(x)

        out = self.conv1(x)
        out = F.max_pool2d(out, 2)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.max_pool2d(out, 2)
        out = F.relu(out)

        out = out.view(b_dx, -1)

        out = self.fc1(out)
        out = F.relu(out)

        out = self.fc2(out)

        return out

    def _init_bbone(self, bbone_weights):
        weights = torch.load(bbone_weights)
        self.denoiser.load_state_dict(weights['model'])

        # freeze layers for the denoiser
        for module in self.denoiser.modules():
            if isinstance(module, nn.Conv2d):
                print('layer frozen.', end='\r')
                for parameters in module.parameters():
                    parameters.requires_grad = False
        print()
