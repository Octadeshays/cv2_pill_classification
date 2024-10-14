import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from PIL import Image
from torchvision import transforms
import torch.optim as optim
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

def get_device():
        # Set device and set to CUDA if GPU is available
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        return device

def train_siamese_network_V2(model, dataloader, optimizer = None, criterion = None, metric = None, epochs = 5, show_bar = True, device = None, name = 'siamese', checkpoint_path = './/checkpoints'):
    
    if(device is None):
        device = get_device()

    model.to(device)

    if(optimizer is None): optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay= 0.001)
    if(criterion is None): criterion = torch.nn.BCEWithLogitsLoss()

    epoch_t_loss = []
    epoch_v_loss = []
    epoch_t_acc = []
    epoch_v_acc = []
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_acc = [], []
        bar = tqdm(dataloader['train'], disable = not show_bar)
        for batch in bar:
            img1, img2, y = batch
            img1, img2, y = img1.to(device), img2.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(img1,img2)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # Metrics
            if(metric is not None): acc = metric(y_hat, y)
            else: acc = (y == (F.sigmoid(y_hat) > 0.5).float()).sum().item() / len(y)
            train_acc.append(acc)
            bar.set_description(f"loss {np.mean(train_loss):.5f} acc {np.mean(train_acc):.5f}")

        epoch_t_loss.append(np.mean(train_loss))
        epoch_t_acc.append(np.mean(train_acc))

        bar = tqdm(dataloader['val'],disable = not show_bar)
        val_loss, val_acc = [], []        
        model.eval()
        with torch.no_grad():
            for batch in bar:
                img1, img2, y = batch
                img1, img2, y = img1.to(device), img2.to(device), y.to(device)
                y_hat = model(img1,img2)
                loss = criterion(y_hat, y)
                val_loss.append(loss.item())
                if(metric is not None): acc = metric(y_hat, y)
                else: acc = (y == (F.sigmoid(y_hat) > 0.5).float()).sum().item() / len(y)
                val_acc.append(acc)
                bar.set_description(f"val_loss {np.mean(val_loss):.5f} val_acc {np.mean(val_acc):.5f}")
        
        if(show_bar):
            print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} val_loss {np.mean(val_loss):.5f} acc {np.mean(train_acc):.5f} val_acc {np.mean(val_acc):.5f}")

        epoch_v_loss.append(np.mean(val_loss))
        epoch_v_acc.append(np.mean(val_acc))
        if(epoch_v_acc[-1] > best_val_acc):
            best_val_acc = epoch_v_acc[-1]
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },f'{checkpoint_path}//{name}.ckpt')
            print(f"Best model saved with Validation Accuracy: {best_val_acc:.4f}")

    return {
        "train_loss": epoch_t_loss,
        "validation_loss": epoch_v_loss,
        "train_accuracy": epoch_t_acc,
        "validation_accuracy": epoch_v_acc
    }

def show_graph(results):
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].plot(results['train_loss'], label="Train loss")
        ax[0].plot(results['validation_loss'], label="Validation loss")
        ax[0].legend()
        ax[0].set_title("Loss by epoch")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("CE")

        ax[1].plot(results['train_accuracy'], label="Train accuracy")
        ax[1].plot(results['validation_accuracy'], label="Validation accuracy")
        ax[1].legend()
        ax[1].set_title("Scores")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("accuracy")
        plt.show()

def test_random_sample(test_loader, model, device = None):
    # Poner el modelo en modo evaluación
    if(device is None):
        device = get_device()
    
    model.to(device)
    model.eval()
    
    # Seleccionar imagen random
    test_iter = iter(test_loader)
    images1, images2, labels = next(test_iter)  # Toma el primer batch del loader
    idx = random.randint(0, len(labels) - 1)

    img1, img2, label = images1[idx].unsqueeze(0), images2[idx].unsqueeze(0), labels[idx].unsqueeze(0)

    img1, img2, label = img1.to(device), img2.to(device), label.to(device)

    with torch.no_grad():
        output = model(img1, img2)
        prediction = (F.sigmoid(output) > 0.5).float()  # Binarizar la predicción

    # Mostrar las imágenes y los resultados
    img1_np = img1.squeeze().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.squeeze().cpu().numpy().transpose(1, 2, 0)

    plt.figure(figsize=(8,4))

    # Mostrar la primera imagen
    plt.subplot(1, 2, 1)
    plt.imshow(img1_np)
    plt.title("Image 1")

    # Mostrar la segunda imagen
    plt.subplot(1, 2, 2)
    plt.imshow(img2_np)
    plt.title("Image 2")

    plt.show()

    # Imprimir la etiqueta verdadera y la predicción
    print(f"True Label: {label.item()}")
    print(f"Predicted Label: {prediction.item()}")

# Función para realizar inferencia y calcular métricas
def evaluate_model(test_loader, model, device = None):
    if(device is None):
        device = get_device()
    model.to(device)
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            # Realizar inferencia
            output = model(img1, img2)
            predictions = (F.sigmoid(output) > 0.5).float()  # Binarizar las predicciones

            # Guardar etiquetas y predicciones
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Calcular las métricas
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=1)
    recall = recall_score(all_labels, all_predictions, zero_division=1)

    # Calcular matriz de confusión
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy, precision, recall, conf_matrix

def test_sample(model, im_ref_path, im_path, input_size=(105,105), device = None):
    if(device is None):
        device = get_device()
    im_ref = Image.open(im_ref_path)
    im_ref = im_ref.resize(input_size).convert("RGB")
    im = Image.open(im_path)
    im = im.resize(input_size).convert("RGB")
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(im_ref)
    ax[1].imshow(im)
    model.to(device)
    model.eval()
    im_ref, im = transforms.functional.to_tensor(im_ref).unsqueeze(0).to(device), transforms.functional.to_tensor(im).unsqueeze(0).to(device)
    res = model(im_ref,im)
    print("Predicted class is", (F.sigmoid(res) > 0.5).float().item())

    return {"Label": (F.sigmoid(res) > 0.5).float().item(),
            "Probability": F.sigmoid(res).item()
            }

