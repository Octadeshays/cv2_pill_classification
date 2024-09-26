import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import random
import os


class SiameseDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        '''
        Pensar como construir dataset que te devuelva un par de imagenes
        '''

    def __len__(self):
        return sum([len(self.class_dict[class_name]) for class_name in self.classes])

    def __getitem__(self, idx):
        'Para cada idx devolver un par de imagenes del dataset'

        return img1, img2, torch.tensor([label], dtype=torch.float32)