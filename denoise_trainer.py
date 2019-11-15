import math
import os
import sys
# pytorch
import torch
from torch.nn.modules.loss import _Loss
import torch.utils.data as torch_data
# models
from models.dncnn import DnCNN
# dataloaders
from dataset import DnCNNDataset
# stats
import matplotlib.pyplot as plt
import numpy as np
# fancy stuff
from tqdm import tqdm


def train_dncnn(model, optimizer, criterion, device, dataloader):
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



def validate_dncnn(model, criterion, device, dataloader):
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



def test_dncnn(model, device, dataloader, debug = False):
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
    if debug:
        for bdx, b in enumerate(outputs):
            for idx , i in enumerate(zip(b[0], b[1], b[2])):
                img = i[0][0].cpu().numpy()
                pred_params = i[1][0].cpu().numpy()
                target_params = i[2][0].cpu().numpy()

                plt.imsave("./results/{}.png".format(idx), img)
                plt.imsave("./results/{}_pred.png".format(idx), pred_params)
                plt.imsave("./results/{}_targ.png".format(idx), target_params)


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


class SSE(_Loss):
    """
    sse = 1/2 * nn.MSELoss (reduced by sum)
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(SSE, self).__init__(
            size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(
            input, target, size_average=None, reduce=None,
            reduction='sum').div_(2)


if __name__=="__main__":
    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()

    epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = DnCNNDataset(count=10000, random_noise=False, noise=2,
        debug=False)
    val_dataset = DnCNNDataset(count=1000, noise=2, random_noise=False,
        debug=False)
    test_dataset = DnCNNDataset(count=1000, noise=2, random_noise=False,
        debug=False)

    train_dataloader = torch_data.DataLoader(train_dataset, num_workers=0, batch_size=32)
    val_dataloader = torch_data.DataLoader(val_dataset, num_workers=0, batch_size=32)
    test_dataloader = torch_data.DataLoader(test_dataset, num_workers=0, batch_size=32)

    model = DnCNN()
    model.to(device)

    print("total parameters: {}".format(total_parameters(model)))

    optimizer = torch.optim.Adam(
        lr=0.005, weight_decay=1e-3, params=model.parameters()
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, verbose=True
    )

    criterion = SSE()

    train_meta = []
    for epoch in range(epochs):
        train_loss = train_dncnn(model, optimizer, criterion, device, train_dataloader)
        val_loss = validate_dncnn(model, criterion, device, val_dataloader)
        test_score = test_dncnn(model, device, test_dataloader)

        scheduler.step(val_loss)

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

        torch.save( state,model_save_str )
        np.save("train_meta_denoiser", np.array(train_meta))
