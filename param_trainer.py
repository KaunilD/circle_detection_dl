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

            self._circle_images.append(
                np.expand_dims(img, axis=0)
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
        image, target = sample[0].float(), sample[1].float()
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
            image, target = sample[0].float(), sample[1].float()
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target)

            val_loss += loss.item()
            tbar.set_description('Val loss:    %.3f' % (train_loss / (i + 1)))
    return val_loss


def test(model, device, dataloader):
    model.eval()
    val_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    outputs = []
    ious = []
    with torch.no_grad():
        for i, sample in enumerate(tbar):

            image = sample[0].float()
            image = image.to(device)
            outputs.append([sample[0], model(image), sample[1]])

    for bdx, b in enumerate(outputs):
        for idx , i in enumerate(zip(b[0], b[1], b[2])):
            img = i[0].cpu().numpy()
            pred_params = i[1].cpu().numpy()
            pred_params = [
                pred_params[0]*100+100,
                pred_params[1]*100+100,
                pred_params[2]*40+10

            ]
            target_params = i[2].cpu().numpy()
            target_params = [
                target_params[0]*100+100,
                target_params[1]*100+100,
                target_params[2]*40+10

            ]

            ious.append(cvp_utils.iou(target_params, pred_params))
    ious = np.asarray(ious)
    return np.mean(ious > 0.7)


if __name__=="__main__":
    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()
    """
    """
    epochs = 400
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
        noise = 0,
        random_noise = False,
        debug=False
    )


    train_dataloader = torch_data.DataLoader(train_dataset, num_workers=0, batch_size=32)
    val_dataloader = torch_data.DataLoader(val_dataset, num_workers=0, batch_size=32)
    test_dataloader = torch_data.DataLoader(test_dataset, num_workers=0, batch_size=32)


    model = CDNet(
        in_planes = 1,
        bbone = DnCNN(),
    )
    checkpoint = torch.load('./results/models/cdnet-9.pth')
    model.load_state_dict(checkpoint['model'])

    model._init_bbone('./results/models/dncnn-70.pth')
    model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Total parameters: {}".format(
        sum([np.prod(p.size()) for p in model_parameters]))
    )

    optimizer = torch.optim.Adam(
        lr=0.005,
        weight_decay=1e-3,
        params=filter(lambda p: p.requires_grad, model.parameters())
    )
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.000005

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, verbose=True
    )

    criterion = nn.MSELoss()

    train_meta = []
    for epoch in range(epochs):
        #train_loss = train(model, optimizer, criterion, device, train_dataloader)
        #val_loss = validate(model, criterion, device, val_dataloader)
        test_score = test(model, device, test_dataloader)
        print(test_score)
        """
        scheduler.step(test_score)

        print(epoch, train_loss, val_loss, test_score)

        train_meta.append(
            [train_loss, val_loss, test_score]
        )

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        model_save_str = './results/models/{}-{}.{}'.format(
            model.name, epoch, "pth"
        )

        torch.save(state, model_save_str)
        np.save("train_meta_param", np.array(train_meta))
        """
