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


class CIRCLEDataset(torch_data.Dataset):
    def __init__(self, count=1000, noise = 1, random_noise = True, debug = False, transform=None):
        self._deubg = debug
        self._count = count
        self._noise = noise
        self._random_noise = random_noise
        self._circle_images, self._circle_params = [], []

        self.transform = transform

        self.create_dataset()

    def create_dataset(self):
        for i in range(self._count):
            # size, radius, noise
            noise = np.random.uniform(0, self._noise) if self._random_noise else self._noise

            params, img, img_noise = cvp_utils.noisy_circle(200, 50, noise)
            # normalize and add a channel axis
            img = (img - np.min(img))/(np.max(img) - np.min(img))
            img_noise = (img_noise - np.min(img_noise))/(np.max(img_noise) - np.min(img_noise))

            self._circle_images.append(
                [
                    np.expand_dims(img_noise, axis=0),
                    np.expand_dims(img, axis=0)
                ]
            )

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


def train(model, optimizer, criterion, device, dataloader):
    model.train()
    train_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    for i, sample in enumerate(tbar):
        image, target = sample[0][0].float(), sample[0][1].float()
        image, target = image.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        tbar.set_description('Train loss:  %.3f' % (train_loss / (i + 1)))
    return train_loss

def validate(model, criterion, device, dataloader):
    model.eval()
    val_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image, target = sample[0][0].float(), sample[0][1].float()
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target)

            val_loss += loss.item()
            tbar.set_description('Val loss:    %.3f' % (train_loss / (i + 1)))
    return val_loss


def test(model, device, dataloader, debug = False):
    model.eval()
    val_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    outputs = []
    ious = []
    with torch.no_grad():
        for i, sample in enumerate(tbar):

            image = sample[0][0].float()
            image = image.to(device)
            outputs.append([sample[0][0], model(image), sample[0][1]])
    if debug:
        for bdx, b in enumerate(outputs):
            for idx , i in enumerate(zip(b[0], b[1], b[2])):
                img = i[0][0].cpu().numpy()
                pred_params = i[1][0].cpu().numpy()
                target_params = i[2][0].cpu().numpy()

                plt.imsave("./results/{}.png".format(idx), img)
                plt.imsave("./results/{}_pred.png".format(idx), pred_params)
                plt.imsave("./results/{}_targ.png".format(idx), target_params)
            

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    sse = 1/2 * nn.MSELoss (reduced by sum)
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(
            input, target, size_average=None, reduce=None, reduction='sum'
        ).div_(2)


if __name__=="__main__":
    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()
    """
    """
    epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = CIRCLEDataset(
        count = 10000,
        random_noise = False,
        noise = 2,
        debug=False
    )
    val_dataset = CIRCLEDataset(
        count = 1000,
        noise = 2,
        random_noise = False,
        debug=False
    )

    test_dataset = CIRCLEDataset(
        count = 1000,
        noise = 2,
        random_noise = False,
        debug=False
    )


    train_dataloader = torch_data.DataLoader(train_dataset, num_workers=0, batch_size=32)
    val_dataloader = torch_data.DataLoader(val_dataset, num_workers=0, batch_size=32)
    test_dataloader = torch_data.DataLoader(test_dataset, num_workers=0, batch_size=32)

    model = DnCNN()
    model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    print("Total parameters: {}".format(
        sum([np.prod(p.size()) for p in model_parameters]))
    )

    optimizer = torch.optim.Adam(
        lr=0.005,
        weight_decay=1e-3,
        params=model.parameters()
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, verbose=True
    )

    criterion = sum_squared_error()
    train_meta = []
    for epoch in range(epochs):
        train_loss = train(model, optimizer, criterion, device, train_dataloader)
        val_loss = validate(model, criterion, device, val_dataloader)

        counter = 0

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        test_score = test(model, device, test_dataloader)
        train_meta.append(
            [train_loss, val_loss, test_score]
        )

        model_save_str = './results/models/{}-{}.{}'.format(
            model.name, epoch, "pth"
        )
        torch.save(
            state,
            model_save_str
        )

        np.save("train_meta_denoiser", np.array(train_meta))

        scheduler.step(val_loss)
        print(epoch, train_loss, val_loss, test_score)
