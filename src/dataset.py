import os
import random
import math
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class SiameseDataset(Dataset):

    """
    Clase para crear un dataset de siamesas que genera pares de imágenes
    etiquetadas como similares o diferentes.

    Esta clase permite generar combinaciones positivas y negativas de imágenes,
    garantizando que no se repitan pares (independientemente del orden) y
    limitando el número de veces que una imagen puede ser seleccionada.

    Args:
        image_folder (str): Ruta a la carpeta que contiene las imágenes organizadas por clase.
        transform (callable, optional): Transformaciones a aplicar a las imágenes.
        max_positive_combinations (int, optional): Número máximo de combinaciones positivas permitidas por imagen.
        max_negative_combinations (int, optional): Número máximo de combinaciones negativas permitidas por imagen.

    Attributes:
        pairs (list): Lista de pares de imágenes generadas.
        generated_pairs (set): Conjunto para almacenar pares ya generados (sin importar el orden).
    """

    def __init__(self, image_folder, transform=None, max_positive_combinations=5, max_negative_combinations=5, seed=42, same_class_negatives = 0.0, same_class_transform = None):
        self.image_folder = image_folder
        self.transform = transform
        self.same_class_negatives = max(0.0, min(same_class_negatives, 1.0))
        self.same_class_transform = same_class_transform
        if(same_class_transform is None and same_class_negatives > 0): warnings.warn("No transforms assigned to same class negative samples")

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor()        
            ])
        self.classes = os.listdir(image_folder)
        self.class_dict = {class_name: os.listdir(os.path.join(image_folder, class_name)) for class_name in self.classes}

        # Conjunto para almacenar combinaciones ya utilizadas (sin importar el orden)
        self.generated_pairs = set()

        # Generar todas las combinaciones posibles
        self.pairs = []
        self.generate_combinations(max_positive_combinations, max_negative_combinations)

    def _add_pair(self, img1, img2, label, class_name1, class_name2, is_same_class_negative = False):
        """
        Añade un par al dataset si no ha sido generado previamente.
        """
        if(not is_same_class_negative):
            pair_key = frozenset([img1, img2])  # Usamos frozenset para evitar repetición sin importar el orden
            if pair_key in self.generated_pairs: return
            else: self.generated_pairs.add(pair_key)
        self.pairs.append((img1, img2, class_name1, class_name2, label, is_same_class_negative))

    def generate_combinations(self, max_positive_combinations, max_negative_combinations):
        """
        Genera todas las combinaciones de pares de imágenes, respetando los límites
        de combinaciones positivas y negativas, evitando duplicados por imagen.
        """
        # Generar combinaciones positivas (máx. max_positive_combinations por imagen)
        for class_name, images in self.class_dict.items():
            num_images = len(images)
            for img1 in images:
                # Calcular la cantidad máxima posible de combinaciones positivas para esta imagen
                max_possible_combinations = min(max_positive_combinations, num_images - 1)
                positive_count = 0  # Contador para limitar combinaciones por imagen
                # Generar combinaciones positivas mientras haya posibilidades
                while positive_count < max_possible_combinations:
                    img2 = random.choice(images)
                    if img1 != img2:  # Evitar pares con la misma imagen
                        self._add_pair(img1, img2, 1, class_name, class_name)  # Etiqueta 1 = Positiva
                        positive_count += 1
        
        # Generar combinaciones negativas (máx. max_negative_combinations por imagen)
        for class_name1, images1 in self.class_dict.items():
            for img1 in images1:
                negative_count = 0  # Contador para limitar combinaciones por imagen
                while negative_count < math.ceil(max_negative_combinations*(1 - self.same_class_negatives)):
                    # Seleccionar una clase diferente para generar el par negativo
                    class_name2 = random.choice([c for c in self.classes if c != class_name1])
                    img2 = random.choice(self.class_dict[class_name2])
                    self._add_pair(img1, img2, 0, class_name1, class_name2)  # Etiqueta 0 = Negativa
                    negative_count += 1
        
        for class_name, images in self.class_dict.items():
            num_images = len(images)
            max_possible_combinations = min(math.floor(max_negative_combinations*self.same_class_negatives), num_images - 1)
            for img1 in images:
                negative_count = 0
                while negative_count < max_possible_combinations:
                    img2 = random.choice(images)
                    if img1 != img2:  # Evitar pares con la misma imagen
                        self._add_pair(img1, img2, 0, class_name, class_name,True)
                        negative_count += 1

        random.shuffle(self.pairs)  # Mezclar los pares para asegurar la aleatoriedad

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Obtener los pares pre-generados
        img1_name, img2_name, class_name1, class_name2, label, same_class_negative = self.pairs[idx]
        
        img1_path = os.path.join(self.image_folder, class_name1, img1_name)
        img2_path = os.path.join(self.image_folder, class_name2, img2_name)

        img1 = Image.open(img1_path) 
        img2 = Image.open(img2_path)

        if self.transform:
            img1 = self.transform(img1)
            if(same_class_negative): img2 = self.same_class_transform(img2)
            else: img2 = self.transform(img2)
        
        return img1, img2, torch.tensor([label], dtype=torch.float32)
    
class TorchAlbumentationTransform:
    def __init__(self, transform):
        self.transform = transform
        self.transform.transforms.extend([A.ToFloat(),ToTensorV2()])
    def __transform(self,img):
        img_np = np.array(img)
        aug = self.transform(image=img_np)
        return aug['image']
    def __call__(self, img):
        return self.__transform(img)