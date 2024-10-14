import torch
import torch.nn as nn
from torchinfo import summary

class OneShot(nn.Module):
    def __init__(self):
        super(OneShot, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(3, 64, 10),  # 64@96*96
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2),  # 64@48*48
        nn.Conv2d(64, 128, 7),
        nn.ReLU(),    # 128@42*42
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),   # 128@21*21
        nn.Conv2d(128, 128, 4),
        nn.ReLU(), # 128@18*18
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2), # 128@9*9
        nn.Conv2d(128, 256, 4),
        nn.ReLU(),   # 256@6*6
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class Siamese(nn.Module):

    def __init__(self, backbone, backbone_output_size = 9216):
        super(Siamese, self).__init__()
        self.backbone = backbone
        self.liner = nn.Sequential(nn.Linear(backbone_output_size, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.backbone(x)
        x = torch.flatten(x,1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out
    
def siameseNet(backbone = None, backbone_output_size = 9216):
    if (backbone is None):
        backbone = OneShot()
    model = Siamese(backbone,backbone_output_size)
    print(summary(model))
    return model