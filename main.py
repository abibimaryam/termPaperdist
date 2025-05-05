import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW
from tqdm import tqdm
import timm
import detectors
from transformer import model_transformer,model_resnet
import json
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

print(model_resnet)
print(model_transformer)

# Настройки
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 32
epochs = 3

# Загрузка данных
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
])

train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transforms)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)


    
# Проверяет точность на валидационной
def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


def layerwise_distillation(
    teacher, 
    student, 
    train_loader, 
    val_loader,
    epochs=5,
    lr=1e-3,
    log_file='distillation_log.txt'
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    teacher = teacher.to(device).eval()
    student = student.to(device)
    
    # Открываем файл для записи логов
    with open(log_file, 'w') as f:
        f.write("=== Layerwise Distillation Training Log ===\n")
        f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Epochs per layer: {epochs}\n")
        f.write(f"Learning rate: {lr}\n\n")
    
    # Слои для дистилляции
    teacher_layers = [
        teacher.layer1,
        teacher.layer2,
        teacher.layer3,
        teacher.layer4
    ]
    
    student_layers = [
        student.layer1,
        student.layer2, 
        student.layer3,
        student.layer4
    ]

    for layer_idx, (t_layer, s_layer) in enumerate(zip(teacher_layers, student_layers)):
        layer_log = f"\n=== Training layer {layer_idx+1} ===\n"
        print(layer_log)
        with open(log_file, 'a') as f:
            f.write(layer_log)
        
        # Замораживаем все слои кроме текущего
        for param in student.parameters():
            param.requires_grad = False
        for param in s_layer.parameters():
            param.requires_grad = True
        
        optimizer = AdamW(s_layer.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            student.train()
            total_loss = 0
            
            for inputs, _ in tqdm(train_loader, desc=f"Layer {layer_idx+1} Epoch {epoch+1}"):
                inputs = inputs.to(device)
                
                # Teacher forward pass
                with torch.no_grad():
                    x_teacher = student.stem(inputs)
                    for l in teacher_layers[:layer_idx]:
                        x_teacher = l(x_teacher)
                    teacher_feat = t_layer(x_teacher)
                
                # Student forward pass
                optimizer.zero_grad()
                x_student = student.stem(inputs)
                for l in student_layers[:layer_idx]:
                    x_student = l(x_student)
                student_feat = s_layer(x_student)
                
                # Align dimensions if needed
                if teacher_feat.shape != student_feat.shape:
                    student_feat = F.adaptive_avg_pool2d(student_feat, teacher_feat.shape[2:])
                
                # Compute loss and update
                loss = criterion(student_feat, teacher_feat)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            epoch_log = f"Layer {layer_idx+1} Epoch {epoch+1} Loss: {avg_loss:.4f}\n"
            print(epoch_log)
            with open(log_file, 'a') as f:
                f.write(epoch_log)
    
    # Размораживаем все слои
    for param in student.parameters():
        param.requires_grad = True
    
    # Валидация
    val_acc = validate(student, val_loader)
    final_log = f"\nFinal validation accuracy: {val_acc:.2f}%\n"
    print(final_log)
    with open(log_file, 'a') as f:
        f.write(final_log)
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return student


model_transformer = layerwise_distillation(
    teacher=model_resnet,
    student=model_transformer,
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=epochs
)

torch.save(model_transformer.state_dict(), 'dist_model_transformer.pth')