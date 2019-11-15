import math
import os
import pickle
import sys
# pytorch
import torch
import torch.utils.data as torch_data
# models
from models.cdnet import CDNet
from models.dncnn import DnCNN
# dataloaders
from dataset import CIRCLEDataset
# stats
import matplotlib.pyplot as plt
import numpy as np
# scale
import cv_practical.main as cvp_utils
# fancy stuff
from tqdm import tqdm


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


def total_parameters(model):
    """Get number parameters in a network.

    Args:
        model: A PyTorch nn.Module object.

    Returns:
        num_parameters (int): total parameters in a network.

    """
    model_parameters = filter(
        lambda p: p.requires_grad, model.parameters())

    return sum([np.prod(p.size()) for p in model_parameters])


if __name__=="__main__":
    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()

    epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataset = CIRCLEDataset(count=1000, noise=2, random_noise=False,
        debug=False)

    test_dataloader = torch_data.DataLoader(test_dataset, num_workers=0,
        batch_size=32)

    model = CDNet(
        in_planes = 1,
        bbone = DnCNN(),
    )

    checkpoint = torch.load('./results/models/cdnet-30.pth')
    model.load_state_dict(checkpoint['model'])

    model.to(device)

    print("total parameters: {}".format(total_parameters(model)))

    for epoch in range(epochs):
        test_score = test(model, device, test_dataloader)
        print(test_score)
