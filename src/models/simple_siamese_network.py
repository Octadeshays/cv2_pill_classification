import torch.nn as nn

class SimpleSiameseNetwork(nn.Module):
    def __init__(self):
        super(SimpleSiameseNetwork, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256)
        )

    def forward_one(self, x):
        x = self.convnet(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
