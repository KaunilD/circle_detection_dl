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

        self.fc1   = nn.Linear(1089, 20*20)
        self.fc2   = nn.Linear(20*20, 20)
        self.fc3   = nn.Linear(20, 3)

        self.name = "cdnet"

    def forward(self, x):
        b_dx, c_dx, w, h = x.size()
        out = self.denoiser(x)

        out = F.avg_pool2d(out, 3, stride=3)
        out = F.avg_pool2d(out, 2, stride=2)

        """
        out = self.conv2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = F.max_pool2d(out, 2)
        out = F.relu(out)
        """
        out = out.view(b_dx, -1)

        out = self.fc1(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = F.relu(out)

        out = self.fc3(out)

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
