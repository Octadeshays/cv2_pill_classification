import torch.nn as nn
import torch

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
            nn.MaxPool2d(2, 2),
        )
        
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 12 * 12, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256)
        )

        # Capa para calcular la distancia entre los embeddings
        self.distance_layer = nn.Linear(256, 1)

    def forward_one(self, x):
        x = self.convnet(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        # Pasar la distancia por una capa lineal para calcular la probabilidad
        
        distance = torch.abs(output1 - output2)
        score = self.distance_layer(distance)  # Genera un valor que luego será mapeado entre 0 y 1
        
        # Aplicar sigmoide para obtener una probabilidad entre 0 y 1
        prob = torch.sigmoid(score)
        return prob
        return output1, output2

