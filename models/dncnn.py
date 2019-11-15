import math
import os
import sys
# pytorch
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DnCNN(nn.Module):
    def __init__( self, n_channels=64, image_channels=1,
        kernel_size=3, init_weights = True ):
        super(DnCNN, self).__init__()
        self.name = "dncnn"

        self._kernel_size = 3
        self._n_channels = 64
        self._image_channels = image_channels
        self._padding = 1


        self.conv_b1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=self._n_channels,
            kernel_size=self._kernel_size, padding=self._padding, bias=True
        )
        self.nl_b1 = nn.ReLU( inplace = True )

        self.conv_b2 = nn.Conv2d(
            in_channels = self._n_channels,
            out_channels = self._n_channels,
            kernel_size = self._kernel_size,
            padding = self._padding, bias=False
        )
        self.bn_2 = nn.BatchNorm2d(self._n_channels, eps = 1e-4, momentum = 0.95)
        self.nl_b2 = nn.ReLU( inplace = True )

        self.conv_b3 = nn.Conv2d(
            in_channels = self._n_channels,
            out_channels = self._n_channels,
            kernel_size = self._kernel_size,
            padding = self._padding, bias=False
        )
        self.bn_3 = nn.BatchNorm2d(self._n_channels, eps = 1e-4, momentum = 0.95)
        self.nl_b3 = nn.ReLU( inplace = True )


        self.conv_b4 = nn.Conv2d(
            in_channels = self._n_channels,
            out_channels = self._n_channels,
            kernel_size = self._kernel_size,
            padding = self._padding, bias=False
        )
        self.bn_4 = nn.BatchNorm2d(self._n_channels, eps = 1e-4, momentum = 0.95)
        self.nl_b4 = nn.ReLU( inplace = True )

        self.conv_b5 = nn.Conv2d(
            in_channels = self._n_channels,
            out_channels = self._image_channels,
            kernel_size = self._kernel_size,
            padding = self._padding,
            bias=False
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        y = x

        out = self.conv_b1(x)
        out = self.nl_b1(out)

        out = self.conv_b2(out)
        out = self.bn_2(out)
        out = self.nl_b2(out)

        out = self.conv_b3(out)
        out = self.bn_3(out)
        out = self.nl_b3(out)

        out = self.conv_b4(out)
        out = self.bn_4(out)
        out = self.nl_b4(out)

        out = self.conv_b5(out)

        return y - out

    def _init_weights(self):
        for mdx, module in enumerate(self.modules()):
            if isinstance(module, nn.Conv2d):
                init.orthogonal_(module.weight)
                print('initialized layer: {}'.format(mdx), end='\r')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
        print("initialization complete!")
