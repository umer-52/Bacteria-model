"""
ViT DIBaS Evaluation Script
Evaluates the trained model on a dataset (Validation/Test) and prints detailed metrics.
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------------- CONFIGURATION ----------------
MODEL_NAME = "vit_base_patch16_224"
NUM_CLASSES = 33
BATCH_SIZE = 32


def evaluate_model(data_dir, checkpoint_path, device='cpu', output_dir='eval_results'):
    print(f"[*] Evaluating model on data from: {data_dir}")
    
    # 1. Prepare Data
    # Use standard validation transforms (Resize + CenterCrop) - No augmentation here!
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    if not os.path.exists(data_dir):
        print(f"[!] Error: Data directory '{data_dir}' not found.")
        return

    dataset = datasets.ImageFolder(data_dir, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    class_names = dataset.classes
    print(f"[*] Found {len(dataset)} images across {len(class_names)} classes.")

    # 2. Load Model
    print(f"[*] Loading model weights from: {checkpoint_path}")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle state dict structure (in case it's nested)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
             
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=True)
        print("[+] Weights loaded successfully.")
    else:
        print(f"[!] Error: Checkpoint file '{checkpoint_path}' not found.")
        return

    model.to(device)
    model.eval()

    # 3. Run Inference
    print("[*] Running inference...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            if (i+1) % 5 == 0:
                print(f"   Processed batch {i+1}/{len(loader)}")

    # 4. Calculate Metrics
    print("\n[*] Evaluation Results")
    print("=" * 60)
    
    # Overall Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"[+] Overall Accuracy: {acc*100:.2f}%")
    print("-" * 60)

    # Classification Report (Precision, Recall, F1)
    # This shows how well the model does for EACH specific bacteria
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=class_names)
    print(report_str)

    # 5. Save Visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Report to CSV
    report_df = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(csv_path)
    print(f"[+] Detailed report saved to: {csv_path}")
    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(24, 20))  # Big figure for 33 classes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix (Acc: {acc*100:.2f}%)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"[+] Confusion Matrix saved to: {cm_path}")
    
    print("=" * 60)
    print("[+] Evaluation Complete!")
    
    return acc, report_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ViT DIBaS Model')
    parser.add_argument('--data_dir', required=True, help='Path to folder containing class subfolders (e.g. test set)')
    parser.add_argument('--checkpoint', default='vit_dibas_v2.pth', help='Path to model checkpoint')
    parser.add_argument('--device', default='auto', help='Device to use (cpu/cuda)')
    parser.add_argument('--output_dir', default='eval_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        
    evaluate_model(args.data_dir, args.checkpoint, device, args.output_dir)
