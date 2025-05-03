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


print(model_resnet)
print(model_transformer)

# Настройки
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 32
epochs = 5

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





#  Обертка для получения промежуточных выходов ResNet (учитель)
class ResNetWrapper(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
        self.features = {}
        
        # хуки для слоев ResNet
        self._register_hooks()
    
    def _register_hooks(self):
        layers = [
            self.resnet.conv1,
            self.resnet.layer1,
            self.resnet.layer2, 
            self.resnet.layer3,
            self.resnet.layer4
        ]
        
        def get_feature_hook(layer_name):
            def hook(module, input, output):
                self.features[layer_name] = output
            return hook
        
        self.handles = []
        for i, layer in enumerate(layers):
            self.handles.append(layer.register_forward_hook(get_feature_hook(f'layer_{i}')))
    
    def forward(self, x):
        self.features.clear()
        return self.resnet(x)
    
    def get_features(self):
        return [self.features[f'layer_{i}'] for i in range(5)]

#  Обертка для получения промежуточных выходов Transformer (студент)
class TransformerWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.features = []
        
    def _hook(self, module, input, output):
        self.features.append(output)
    
    def forward(self, x):
        self.features.clear()
        
        #  хуки
        handles = []
        modules = [
            self.transformer.stem[0],
            self.transformer.layer1,
            self.transformer.layer2,
            self.transformer.layer3,
            self.transformer.layer4
        ]
        
        for module in modules:
            handles.append(module.register_forward_hook(self._hook))
        
        output = self.transformer(x)
        
        # Удаляем хуки после forward pass
        for handle in handles:
            handle.remove()
            
        return output
    
    def get_features(self):
        return self.features
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

# # 3. Функция дистилляции
# def distillation(teacher, student, train_loader, val_loader, epochs=10):
#     # Обертки моделей
#     teacher_wrapped = ResNetWrapper(teacher).to(device)
#     student_wrapped = TransformerWrapper(student).to(device)
#     best_acc=0
#     optimizer = AdamW(student.parameters(), lr=1e-3)
#     cls_criterion = nn.CrossEntropyLoss()
    
#     for epoch in range(epochs):
#         student.train()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         for inputs, targets in tqdm(train_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
            
#             # Forward pass учителя
#             with torch.no_grad():
#                 teacher_logits = teacher_wrapped(inputs)
#                 teacher_features = teacher_wrapped.get_features()
            
#             # Forward pass студента
#             optimizer.zero_grad()
#             student_logits = student_wrapped(inputs)
#             student_features = student_wrapped.get_features()
            
#             # Вычисление потерь
#             loss_cls = cls_criterion(student_logits, targets)
            
#             # Feature matching loss
#             loss_feat = 0
#             for t_feat, s_feat in zip(teacher_features, student_features):
#                 # Приводим размерности к совместимым
#                 if t_feat.shape != s_feat.shape:
#                     s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
#                 loss_feat += F.mse_loss(s_feat, t_feat)
            
#             # Общие потери
#             loss = loss_cls + 0.1 * loss_feat 
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             _, predicted = student_logits.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
        
#         train_acc = 100. * correct / total
#         val_acc = validate(student, val_loader)
        
#         print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, '
#               f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(student.state_dict(), 'student_after_distillation.pth')
#             print("Saved new best model!")


# # Запуск дистилляции
# distillation(model_resnet, model_transformer, train_loader, test_loader, epochs=epochs)


def distillation_init(
    teacher, 
    student, 
    train_loader, 
    val_loader, 
    init_epochs=epochs,          
    lr=1e-3,               # Learning rate
    feat_loss_weight=0.1,   # Вес для feature loss
    temp=1.0,              # Температура для дистилляции      
):
    # Переводим модели на нужное устройство
    teacher = teacher.to(device).eval()
    student = student.to(device)
    
    # Обертки для захвата признаков
    teacher_wrapped = ResNetWrapper(teacher)
    student_wrapped = TransformerWrapper(student)
    
    # Оптимизатор и критерии
    optimizer = AdamW(student.parameters(), lr=lr)
    cls_criterion = nn.CrossEntropyLoss()
    kldiv_criterion = nn.KLDivLoss(reduction='batchmean')
    
    best_acc = 0
    best_state = None

    for epoch in range(init_epochs):
        student.train()
        total_loss = 0
        total_samples = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Init Epoch {epoch+1}/{init_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass учителя
            with torch.no_grad():
                teacher_logits = teacher_wrapped(inputs)
                teacher_features = teacher_wrapped.get_features()
                teacher_probs = F.softmax(teacher_logits / temp, dim=1)
            
            # Forward pass студента
            optimizer.zero_grad()
            student_logits = student_wrapped(inputs)
            student_features = student_wrapped.get_features()
            student_probs = F.log_softmax(student_logits / temp, dim=1)
            
            # Вычисление потерь
            # Потеря классификации
            loss_cls = cls_criterion(student_logits, targets)
            
            # Потеря дистилляции 
            loss_distill = kldiv_criterion(student_probs, teacher_probs) * (temp**2)
            
            #  Потеря по признакам
            loss_feat = 0
            for t_feat, s_feat in zip(teacher_features, student_features):
                if t_feat.shape != s_feat.shape:
                    s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
                loss_feat += F.mse_loss(s_feat, t_feat)
            
            # Комбинированная потеря
            loss = loss_cls + loss_distill + feat_loss_weight * loss_feat
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        
        # Валидация
        val_acc = validate(student, val_loader)
        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch+1}/{init_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Сохранение лучшей модели
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = student.state_dict()
            torch.save(best_state, 'student_init_best.pth')
            print(f"Saved best model with acc: {best_acc:.2f}%")
    

    return student


model_transformer = distillation_init(
    teacher=model_resnet,
    student=model_transformer,
    train_loader=train_loader,
    val_loader=test_loader,
)