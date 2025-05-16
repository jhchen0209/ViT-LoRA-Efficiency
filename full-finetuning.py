# vit_full_finetune_pytorch.py
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.model_selection import train_test_split
import pandas as pd


# ----------------------
# Dataset 與預處理
# ----------------------
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]


def load_data(data_dir):
    image_paths = []
    labels = []
    class_names = set()
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            class_name = "_".join(filename.split("_")[:-1])
            image_paths.append(os.path.join(data_dir, filename))
            class_names.add(class_name)
    class_names = sorted(list(class_names))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    for path in image_paths:
        class_name = "_".join(os.path.basename(path).split("_")[:-1])
        labels.append(class_to_idx[class_name])
    return image_paths, labels, class_names


# ----------------------
# 訓練與驗證
# ----------------------
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=x, labels=y)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (outputs.logits.argmax(dim=-1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(pixel_values=x, labels=y)
            loss = outputs.loss
            total_loss += loss.item() * x.size(0)
            correct += (outputs.logits.argmax(dim=-1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# ----------------------
# 主程式邏輯
# ----------------------
def main():
    data_dir = "./DATA/images"
    image_paths, labels, class_names = load_data(data_dir)
    train_idx, test_idx = train_test_split(range(len(image_paths)), test_size=0.2, stratify=labels, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    train_set = ImageDataset([image_paths[i] for i in train_idx], [labels[i] for i in train_idx], transform)
    test_set = ImageDataset([image_paths[i] for i in test_idx], [labels[i] for i in test_idx], transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained("facebook/dino-vits16", num_labels=len(class_names), ignore_mismatched_sizes=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start_time = time.time()
    for epoch in range(20):
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"訓練完成，總耗時: {elapsed:.2f} 秒")

    # 額外輸出模型參數資訊
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== 模型參數統計 ===")
    print(f"總參數數量: {total_params:,}")
    print(f"可訓練參數: {trainable_params:,}")
    print(f"可訓練比例: {100 * trainable_params / total_params:.2f}%")


if __name__ == "__main__":
    main()