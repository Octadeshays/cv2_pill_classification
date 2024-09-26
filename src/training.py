import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.loss import ContrastiveLoss

def train_siamese_network(model, train_loader, num_epochs, device, lr=0.001):
    """
    Función para entrenar una red siamesa.

    Args:
    - model: La red siamesa a entrenar.
    - train_loader: El DataLoader para el conjunto de entrenamiento.
    - num_epochs: Número de épocas para entrenar.
    - device: Dispositivo donde correr el entrenamiento ('cuda' o 'cpu').
    - lr: Tasa de aprendizaje para el optimizador.
    """
    model = model.to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            # Adelante
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)

            # Atrás
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    print("Entrenamiento finalizado.")



