# Source: https://www.kaggle.com/yasufuminakama/panda-pytorch-grad-cam

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
import cv2
import PIL

print(PIL.__version__)


class SaveFeatures:
    """Extract pretrained activations"""

    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()

    def remove(self):
        self.hook.remove()


def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(
        feature_conv[
            0,
            :,
            :,
        ].reshape((nc, h * w))
    )
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img


def plotGradCAM(
    model,
    final_conv,
    fc_params,
    train_loader,
    row=1,
    col=8,
    img_size=256,
    device="cpu",
    original=False,
):
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    # save activated_features from conv
    activated_features = SaveFeatures(final_conv)
    # save weight from fc
    weight = np.squeeze(fc_params[0].cpu().data.numpy())
    # original images
    if original:
        fig = plt.figure(figsize=(20, 15))
        for i, (img, target, org_img) in enumerate(train_loader):
            output = model(img.to(device))
            pred_idx = output.to("cpu").numpy().argmax(1)
            cur_images = org_img.numpy().transpose((0, 2, 3, 1))
            ax = fig.add_subplot(row, col, i + 1, xticks=[], yticks=[])
            plt.imshow(cv2.cvtColor(cur_images[0], cv2.COLOR_BGR2RGB))
            ax.set_title("Label:%d, Predict:%d" % (target, pred_idx), fontsize=14)
            if i == row * col - 1:
                break
        plt.show()
    # heatmap images
    fig = plt.figure(figsize=(20, 15))
    for i, (img, target, _) in enumerate(train_loader):
        output = model(img.to(device))
        pred_idx = output.to("cpu").numpy().argmax(1)
        cur_images = img.cpu().numpy().transpose((0, 2, 3, 1))
        heatmap = getCAM(activated_features.features, weight, pred_idx)
        ax = fig.add_subplot(row, col, i + 1, xticks=[], yticks=[])
        plt.imshow(cv2.cvtColor(cur_images[0], cv2.COLOR_BGR2RGB))
        plt.imshow(
            cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_LINEAR),
            alpha=0.4,
            cmap="jet",
        )
        ax.set_title("Label:%d, Predict:%d" % (target, pred_idx), fontsize=14)
        if i == row * col - 1:
            break
    plt.show()
