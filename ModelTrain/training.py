import os
import numpy as np
from PIL import Image
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Dataset class
class MegaAgeDataset(Dataset):
    def __init__(self, image_dir, dis_file, age_file, gender_file, name_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.distributions = np.loadtxt(dis_file, dtype=float)
        self.age_labels = np.loadtxt(age_file, dtype=int)
        self.gender_labels = np.loadtxt(gender_file, dtype=int)
        self.names = np.loadtxt(name_file, dtype=str)

        valid_indices = self.gender_labels != 2
        self.age_labels = self.age_labels[valid_indices]
        self.gender_labels = self.gender_labels[valid_indices]
        self.names = self.names[valid_indices]
        self.distributions = self.distributions[valid_indices]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        distribution = torch.tensor(self.distributions[idx], dtype=torch.float32)
        age_label = torch.tensor(self.age_labels[idx], dtype=torch.long)
        gender_label = torch.tensor(self.gender_labels[idx], dtype=torch.long)

        return image, distribution, age_label, gender_label

# Separate Models
class AgeModel(nn.Module):
    def __init__(self):
        super(AgeModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.age_head = nn.Linear(in_features, 70)

    def forward(self, x):
        features = self.backbone(x)
        return self.age_head(features)

class GenderModel(nn.Module):
    def __init__(self):
        super(GenderModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.gender_head = nn.Linear(in_features, 2)

    def forward(self, x):
        features = self.backbone(x)
        return self.gender_head(features)

# Training Age Model

def train_age_model(model, train_loader, test_loader, optimizer, scheduler, criterion, device, use_kl=False, early_stop=5):
    best_model_wts = None
    best_test_loss = float('inf')
    stop_count = 0

    for epoch in range(50):
        model.train()
        running_loss = 0.0

        for images, dists, ages, _ in train_loader:
            images = images.to(device)
            dists = dists.to(device)
            ages = ages.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            if use_kl:
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                loss = criterion(log_probs, dists)
            else:
                expected_ages = torch.sum(torch.softmax(outputs, dim=1) * torch.arange(70, device=device), dim=1)
                loss = criterion(expected_ages, ages.float())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        avg_test_loss = evaluate_age(model, test_loader, criterion, device, use_kl)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss <= best_test_loss:
            best_model_wts = model.state_dict()
            best_test_loss = avg_test_loss
            stop_count = 0
        else:
            stop_count += 1
        if stop_count >= early_stop:
            print("Early stopping.")
            break

    model.load_state_dict(best_model_wts)
    os.makedirs("gpt_age", exist_ok=True)
    torch.save(model.state_dict(), "gpt_age/best_model.pth")

# Training Gender Model

def train_gender_model(model, train_loader, test_loader, optimizer, scheduler, criterion, device, early_stop=5):
    best_model_wts = None
    best_test_acc = 0.0
    stop_count = 0

    for epoch in range(50):
        model.train()
        running_acc = 0.0

        for images, _, _, genders in train_loader:
            images = images.to(device)
            genders = genders.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, genders)
            loss.backward()
            optimizer.step()

            running_acc += (outputs.argmax(1) == genders).float().mean().item()

        scheduler.step()

        avg_train_acc = running_acc / len(train_loader)
        avg_test_acc = evaluate_gender(model, test_loader, device)
        print(f"Epoch {epoch+1}: Train Acc: {avg_train_acc:.4f}, Test Acc: {avg_test_acc:.4f}")

        if avg_test_acc >= best_test_acc:
            best_model_wts = model.state_dict()
            best_test_acc = avg_test_acc
            stop_count = 0
        else:
            stop_count += 1
        if stop_count >= early_stop:
            print("Early stopping.")
            break

    model.load_state_dict(best_model_wts)
    os.makedirs("gpt_gender", exist_ok=True)
    torch.save(model.state_dict(), "gpt_gender/best_model.pth")

# Evaluation Functions

def evaluate_age(model, dataloader, criterion, device, use_kl):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, dists, ages, _ in dataloader:
            images = images.to(device)
            dists = dists.to(device)
            ages = ages.to(device)
            outputs = model(images)
            if use_kl:
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                loss = criterion(log_probs, dists)
            else:
                expected_ages = torch.sum(torch.softmax(outputs, dim=1) * torch.arange(70, device=device), dim=1)
                loss = criterion(expected_ages, ages.float())
            total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_gender(model, dataloader, device):
    model.eval()
    total_acc = 0.0
    with torch.no_grad():
        for images, _, _, genders in dataloader:
            images = images.to(device)
            genders = genders.to(device)
            outputs = model(images)
            total_acc += (outputs.argmax(1) == genders).float().mean().item()
    return total_acc / len(dataloader)

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['kl', 'mse', 'gender'], default='mse')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = MegaAgeDataset(
        image_dir="megaage_processed/train",
        dis_file="megaage/list/train_dis.txt",
        age_file="megaage/list/train_age.txt",
        gender_file="megaage/list/train_gender.txt",
        name_file="megaage/list/train_name.txt",
        transform=transform
    )

    test_dataset = MegaAgeDataset(
        image_dir="megaage_processed/test",
        dis_file="megaage/list/test_dis.txt",
        age_file="megaage/list/test_age.txt",
        gender_file="megaage/list/test_gender.txt",
        name_file="megaage/list/test_name.txt",
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    if args.mode == 'gender':
        model = GenderModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion_gender = nn.CrossEntropyLoss()
        print("Starting Gender model training...")
        train_gender_model(model, train_loader, test_loader, optimizer, scheduler, criterion_gender, device)

    else:
        model = AgeModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        if args.mode == 'kl':
            criterion = nn.KLDivLoss(reduction='batchmean')
            print("Starting Age model KL training...")
            train_age_model(model, train_loader, test_loader, optimizer, scheduler, criterion, device, use_kl=True)
        else:
            criterion = nn.MSELoss()
            print("Starting Age model MSE training...")
            train_age_model(model, train_loader, test_loader, optimizer, scheduler, criterion, device)

    print("Training complete.")
