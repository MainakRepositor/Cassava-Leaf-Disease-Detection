# import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter


# resnext50_32x4d model
class resnext50_32x4d(nn.Module):
    def __init__(self, model_name="resnext50_32x4d", targets=5, pretrained=False):
        super().__init__()
        self.model = models.resnext50_32x4d()
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, targets)

    def forward(self, x):
        x = self.model(x)
        return x


# Vision transformer model
# class ViT_model(nn.Module):
#    def __init__(self, model_name='vit_base_patch16_384', targets=5, pretrained=False):
#        super().__init__()
#        self.model = timm.create_model(model_name, pretrained=pretrained)
#        n_features = self.model.head.in_features
#        self.model.head = nn.Linear(n_features, targets)

#    def forward(self, x):
#        x = self.model(x)
#        return x

# tf_efficientnet_b3_ns model
# class efficientnet_b3_ns(nn.Module):
#    def __init__(self, model_name='tf_efficientnet_b3_ns',targets=5, pretrained=False):
#        super().__init__()
#        self.model = timm.create_model(model_name, pretrained=pretrained)
#        n_features = self.model.fc.in_features
#        self.model.fc = nn.Linear(n_features, targets)

#   def forward(self, x):
#        x = self.model(x)
#        return x
