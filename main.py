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
seed = 42
batch_size = 32
epochs = 50
torch.manual_seed(seed)

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



# # Модель для дистилляции
# class DistillationWrapper(nn.Module):
#     def __init__(self, teacher, student, temperature=3.0, alpha=0.5):
#         super().__init__()
#         self.teacher = teacher
#         self.student = student
#         self.temperature = temperature
#         self.alpha = alpha
        
#         # Замораживаем учителя
#         for param in self.teacher.parameters():
#             param.requires_grad = False
#         self.teacher.eval()

#     def forward(self, x):
#         student_logits = self.student(x)
#         with torch.no_grad():
#             teacher_logits = self.teacher(x)
#         return student_logits, teacher_logits

#     def compute_loss(self, student_logits, teacher_logits, targets):
#         # Loss студента
#         student_loss = F.cross_entropy(student_logits, targets)
        
#         # Дистилляционный loss
#         distillation_loss = F.kl_div(
#             F.log_softmax(student_logits / self.temperature, dim=1),
#             F.softmax(teacher_logits / self.temperature, dim=1),
#             reduction='batchmean'
#         ) * (self.temperature ** 2)
        
#         # Комбинированный loss
#         return (1 - self.alpha) * student_loss + self.alpha * distillation_loss

# # Инициализация моделей
# teacher = model_resnet
# student = model_transformer
# model = DistillationWrapper(teacher, student).to(device)

# optimizer = AdamW(student.parameters(), lr=3e-4, weight_decay=0.01)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# # Функции для обучения
# def accuracy(output, target):
#     with torch.no_grad():
#         pred = output.argmax(dim=1)
#         correct = (pred == target).float().sum()
#         return correct / len(target)

# def train_epoch(model, loader, optimizer):
#     model.train()
#     total_loss, total_acc = 0, 0
    
#     for x, y in tqdm(loader, desc="Training"):
#         x, y = x.to(device), y.to(device)
        
#         optimizer.zero_grad()
#         student_logits, teacher_logits = model(x)
#         loss = model.compute_loss(student_logits, teacher_logits, y)
        
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
#         total_acc += accuracy(student_logits, y).item()
    
#     return total_loss / len(loader), total_acc / len(loader)

# def validate(model, loader):
#     model.eval()
#     total_loss, total_acc = 0, 0
    
#     with torch.no_grad():
#         for x, y in tqdm(loader, desc="Validation"):
#             x, y = x.to(device), y.to(device)
#             student_logits, teacher_logits = model(x)
#             loss = model.compute_loss(student_logits, teacher_logits, y)
            
#             total_loss += loss.item()
#             total_acc += accuracy(student_logits, y).item()
    
#     return total_loss / len(loader), total_acc / len(loader)



# log_data = []

# # Обучение
# best_acc = 0
# for epoch in range(epochs):
#     train_loss, train_acc = train_epoch(model, train_loader, optimizer)
#     val_loss, val_acc = validate(model, test_loader)
#     scheduler.step()
    
#     print(f"Epoch {epoch+1}/{epochs}:")
#     print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
#     print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
#     log_data.append({
#         "epoch": epoch + 1,
#         "train_loss": round(train_loss, 4),
#         "train_acc": round(train_acc, 4),
#         "val_loss": round(val_loss, 4),
#         "val_acc": round(val_acc, 4)
#     })


#     with open("student.txt", "w") as f:
#         json.dump(log_data, f, indent=4)

#     if val_acc > best_acc:
#         best_acc = val_acc
#         torch.save(student.state_dict(), 'best_student.pth')
#         print("Saved new best model!")

# print(f"Best Validation Accuracy: {best_acc:.4f}")

# 1. Обертка для получения промежуточных выходов ResNet (учитель)
class ResNetWrapper(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
        self.features = {}
        
        # Регистрируем хуки для слоев ResNet
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

# 2. Обертка для получения промежуточных выходов Transformer (студент)
class TransformerWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.features = []
        
    def _hook(self, module, input, output):
        self.features.append(output)
    
    def forward(self, x):
        self.features.clear()
        
        # Регистрируем временные хуки
        handles = []
        modules = [
            self.transformer.stem,
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

# 3. Функция дистилляции
def train_distillation(teacher, student, train_loader, val_loader, epochs=10):
    # Обертки моделей
    teacher_wrapped = ResNetWrapper(teacher).to(device)
    student_wrapped = TransformerWrapper(student).to(device)
    
    optimizer = AdamW(student.parameters(), lr=1e-3)
    cls_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass учителя
            with torch.no_grad():
                teacher_logits = teacher_wrapped(inputs)
                teacher_features = teacher_wrapped.get_features()
            
            # Forward pass студента
            optimizer.zero_grad()
            student_logits = student_wrapped(inputs)
            student_features = student_wrapped.get_features()
            
            # Вычисление потерь
            loss_cls = cls_criterion(student_logits, targets)
            
            # Feature matching loss
            loss_feat = 0
            for t_feat, s_feat in zip(teacher_features, student_features):
                # Приводим размерности к совместимым
                if t_feat.shape != s_feat.shape:
                    s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
                loss_feat += F.mse_loss(s_feat, t_feat)
            
            # Общие потери
            loss = loss_cls + 0.1 * loss_feat  # Коэффициент можно регулировать
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        val_acc = validate(student, val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')


# Запуск дистилляции
train_distillation(model_resnet, model_transformer, train_loader, test_loader, epochs=epochs)
torch.save(model_transformer.state_dict(), 'student_after_distillation.pth')
