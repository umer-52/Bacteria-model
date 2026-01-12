"""
ViT DIBaS Training Script (Kaggle Compatible)
Fine-tunes vit_base_patch16_224 on bacterial colony images.
Supports merging original DIBaS dataset with custom images.
"""

import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from IPython.display import FileLink
from PIL import Image

# ---------------- 1. CONFIGURATION ----------------
# Paths
ORIGINAL_DATA_DIR = "/kaggle/input/dibas-bacterial-colony-dataset"
NEW_DATA_DIR = "/kaggle/input/dibas-extra-images/my_new_data"
COMBINED_DIR = "/kaggle/working/full_dataset"
OUTPUT_DIR = "/kaggle/working/finetune_output"

MODEL_NAME = "vit_base_patch16_224"
NUM_CLASSES = 33

# --- TUNING PARAMETERS ---
EPOCHS = 30          # Increased from 10 to 30
BATCH_SIZE = 32      # Keep 32 (Safe). Try 64 if you want faster speed.
LR = 3e-4            # Slightly lower than 1e-3 for smoother fine-tuning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- 2. MERGE DATASETS ----------------
def setup_combined_dataset():
    if os.path.exists(COMBINED_DIR):
        print(f"[+] Combined folder {COMBINED_DIR} already exists. Using it.")
        return

    print("[*] Building Combined Dataset...")
    # Copy Original
    if os.path.exists(ORIGINAL_DATA_DIR):
        shutil.copytree(ORIGINAL_DATA_DIR, COMBINED_DIR)
    
    # Merge New Data
    if os.path.exists(NEW_DATA_DIR):
        added = 0
        for root, dirs, files in os.walk(NEW_DATA_DIR):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    src = os.path.join(root, file)
                    cls = os.path.basename(root)
                    dst = os.path.join(COMBINED_DIR, cls)
                    os.makedirs(dst, exist_ok=True)
                    shutil.copy(src, os.path.join(dst, file))
                    added += 1
        print(f"[+] Added {added} new images.")
    else:
        print("[!] New data folder not found, using original only.")


# ---------------- 3. CLEANUP (Fixes the UnidentifiedImageError) ----------------
def clean_dataset(root_dir):
    print("[*] Checking for corrupt images...")
    removed = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                os.remove(path)
                removed += 1
    print(f"[+] Removed {removed} corrupt files.")


# ---------------- 4. TRAINING ----------------
def train():
    setup_combined_dataset()
    clean_dataset(COMBINED_DIR)
    
    # Setup Data
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(COMBINED_DIR, transform=tf)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"[*] Loaded {len(dataset)} images. Starting {EPOCHS} epochs...")

    # Model
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    model.to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_acc = 0.0  # Track accuracy instead of loss for saving
    
    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        loop = tqdm(loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, pred = out.max(1)
            total += lbls.size(0)
            correct += pred.eq(lbls).sum().item()
            
            loop.set_postfix(acc=f"{100.*correct/total:.1f}%", loss=f"{loss.item():.3f}")
            
        epoch_acc = 100. * correct / total
        
        # SAVE LOGIC: Save if accuracy improves
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/vit_dibas_best.pth")
            print(f"   [!] New Best! Acc: {best_acc:.2f}%")

    print("\n[+] Training Complete.")


# ---------------- 5. DOWNLOAD FIX ----------------
def get_download_link():
    # Move file to root to avoid path issues
    src = f"{OUTPUT_DIR}/vit_dibas_best.pth"
    dst = "/kaggle/working/vit_dibas_best.pth"
    
    if os.path.exists(src):
        shutil.copy(src, dst)
        print("[*] Click the link below to download your model:")
        display(FileLink(r'vit_dibas_best.pth'))
    else:
        print("[!] Error: Model file not found. Did training finish?")


# ---------------- 6. MAIN EXECUTION ----------------
if __name__ == "__main__":
    print("=" * 50)
    print("ViT DIBaS Training Script")
    print(f"Model: {MODEL_NAME}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LR}")
    print("=" * 50)
    
    train()
    get_download_link()
