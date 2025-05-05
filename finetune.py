import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torchvision import datasets, transforms
from transformer import model_transformer
from datetime import datetime

# from main import device,validate,train_loader,test_loader,model_transformer

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


def fine_tune(model, train_loader, val_loader, epochs=20, lr=1e-3, log_file='training_log.txt'):
    model.to(device)
    model.train()

    with open(log_file, 'w') as f:
        f.write("=== Fine-Tuning Training Log ===\n")
        f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total epochs: {epochs}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Batch size: {train_loader.batch_size}\n")
        f.write(f"Device: {device}\n\n")

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss/len(train_loader)
        train_acc = 100. * correct / total
        val_acc = validate(model, val_loader)
        scheduler.step()


        log_str = (f'Epoch {epoch+1}/{epochs}\n'
                  f'Loss: {train_loss:.4f}\n'
                  f'Train Accuracy: {train_acc:.2f}%\n'
                  f'Validation Accuracy: {val_acc:.2f}%\n'
                  f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n'
                  f'{"Saved new best model!" if val_acc > best_acc else ""}\n'
                  f'{"-"*50}\n')


        print(log_str)


        with open(log_file, 'a') as f:
            f.write(log_str)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'student_finetuned.pth')

    final_log = (f'\nTraining completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
                f'Best Validation Accuracy: {best_acc:.2f}%\n'
                f'Model saved to: student_finetuned.pth\n')

    with open(log_file, 'a') as f:
        f.write(final_log)
    print(final_log)

    return model

torch.serialization.add_safe_globals([model_transformer])
model_transformer.load_state_dict(torch.load('dist_model_transformer.pth'))
fine_tune(model_transformer, train_loader, test_loader, epochs=20)
