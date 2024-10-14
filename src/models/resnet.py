import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# A recursive function to freeze parameters up to ratio of total parameters, and reset the others.
# Exlusively made for the NoDenseNet class.
def shallow_feature_extract(model, ratio = 0.8):
    total_layers = 0
    for i,children in enumerate(model.children()):
        total_layers+=1
    frozen_layers = (total_layers * ratio)
    for i,children in enumerate(model.children()):
        for module in children.modules():
            if(i < frozen_layers):
                children.requires_grad_(False)
            else:
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    return model

def prunnedResnet18(pretrained = True, freeze_ratio = 0.8, remove_classifier = True):
    if(pretrained): weights = ResNet18_Weights.IMAGENET1K_V1
    else: weights = ResNet18_Weights.DEFAULT
    resnet = resnet18(weights=weights)
    if(freeze_ratio != 0):
        resnet = shallow_feature_extract(resnet,ratio=freeze_ratio)
    if(remove_classifier): resnet.fc = nn.Identity()
    return resnet