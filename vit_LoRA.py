import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.model_selection import train_test_split
import pandas as pd

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, dropout=0.1):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.scaling = lora_alpha / r
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = F.linear(self.dropout(x), self.lora_A)  # shape: [B, *, r]
            lora_out = F.linear(lora_out, self.lora_B)         # shape: [B, *, out_features]
            result += lora_out * self.scaling
        return result



def apply_lora_to_vit(model, target_modules=["attention.query", "attention.value"], r=8, lora_alpha=16, dropout=0.1):
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target):
                parent = model
                subnames = name.split(".")
                for sub in subnames[:-1]:
                    parent = getattr(parent, sub)
                old_layer = getattr(parent, subnames[-1])
                if isinstance(old_layer, nn.Linear):
                    lora_layer = LoRALinear(old_layer.in_features, old_layer.out_features, r, lora_alpha, dropout)
                    lora_layer.weight.data.copy_(old_layer.weight.data)
                    lora_layer.bias.data.copy_(old_layer.bias.data)
                    setattr(parent, subnames[-1], lora_layer)

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
    model = ViTForImageClassification.from_pretrained("facebook/dino-vits16", num_labels=len(class_names))
    apply_lora_to_vit(model)
    for name, param in model.named_parameters():
        if "lora_" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.to(device)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    start_time = time.time()
    for epoch in range(20):
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"訓練完成，總耗時: {elapsed:.2f} 秒")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== 模型參數統計 ===")
    print(f"總參數數量: {total_params:,}")
    print(f"可訓練參數: {trainable_params:,}")
    print(f"可訓練比例: {100 * trainable_params / total_params:.2f}%")

if __name__ == "__main__":
    main()
