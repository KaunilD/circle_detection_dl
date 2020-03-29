import sys
sys.path.append('..')
import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
# pytorch
import torch
# models
from models import cdnet
from models import dncnn


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]



def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)
    img_noise = np.copy(img)
    # Noise
    img_noise += noise * np.random.rand(*img_noise.shape)
    return (row, col, rad), img, img_noise


def find_circle(img):
    img = min_max(img)

    img = torch.tensor(
        np.expand_dims(np.array([img]), axis=0)
    ).float()

    out = model(img).detach().numpy()[0]
    out = [
        out[0]*100+100,
        out[1]*100+100,
        out[2]*40+10
    ]

    return out


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def load_model(model_path):
    model = cdnet.CDNet(
        in_planes = 1,
        bbone=dncnn.DnCNN()
    )

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.to(torch.device("cpu"))
    return model

def main():
    """Added by Kaunil D.
    """
    global model
    model = load_model('../results/models/cdnet-30.pth')

    global min_max
    min_max = lambda a: (a-np.min(a))/(np.max(a) - np.min(a))

    results = []
    for _ in range(1000):
        params, img, _ = noisy_circle(200, 50, 2)
        detected = find_circle(img)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())
