"""
ViT DIBaS Evaluation Script (Metrics & F1 Scores)
Evaluates a trained model on a flat folder of images using a CSV file for ground truth labels.
Generates detailed classification reports (F1, Precision, Recall) and Confusion Matrix.
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ---------------- CONFIGURATION ----------------
# The 33 DIBaS classes in alphabetical order. 
# CRITICAL: This order MUST match the order used during training (ImageFolder default).
CLASS_NAMES = [
    'Acinetobacter.baumanii',
    'Actinomyces.israeli',
    'Bacteroides.fragilis',
    'Bifidobacterium.spp',
    'Candida.albicans',
    'Clostridium.perfringens',
    'Enterococcus.faecalis',
    'Enterococcus.faecium',
    'Escherichia.coli',
    'Fusobacterium.spp',
    'Lactobacillus.casei',
    'Lactobacillus.crispatus',
    'Lactobacillus.delbrueckii',
    'Lactobacillus.gasseri',
    'Lactobacillus.jehnsenii',
    'Lactobacillus.johnsonii',
    'Lactobacillus.paracasei',
    'Lactobacillus.plantarum',
    'Lactobacillus.reuteri',
    'Lactobacillus.rhamnosus',
    'Lactobacillus.salivarius',
    'Listeria.monocytogenes',
    'Micrococcus.spp',
    'Neisseria.gonorrhoeae',
    'Porphyromonas.gingivalis',
    'Propionibacterium.acnes',
    'Proteus.spp',
    'Pseudomonas.aeruginosa',
    'Staphylococcus.aureus',
    'Staphylococcus.epidermidis',
    'Staphylococcus.saprophyticus',
    'Streptococcus.agalactiae',
    'Veionella.spp'
]

MODEL_NAME = "vit_base_patch16_224"
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 32

# ---------------- DATASET CLASS ----------------
class FlatImageDataset(Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load labels
        self.df = pd.read_csv(csv_path)
        # Expecting CSV columns: 'filename', 'label'
        # Adjust column names if necessary based on user's CSV format
        if 'filename' not in self.df.columns or 'label' not in self.df.columns:
            # Fallback: try using first two columns if named differently
            print("[!] Warning: CSV columns 'filename' and 'label' not found. Using first two columns.")
            self.file_col = self.df.columns[0]
            self.label_col = self.df.columns[1]
        else:
            self.file_col = 'filename'
            self.label_col = 'label'
            
        # Filter out images that don't exist
        self.valid_data = []
        missing_count = 0
        
        print(f"[*] Checking dataset integrity...")
        for idx, row in self.df.iterrows():
            fname = row[self.file_col]
            label = row[self.label_col]
            path = os.path.join(data_dir, fname)
            
            if os.path.exists(path):
                if label in CLASS_NAMES:
                    self.valid_data.append((path, CLASS_NAMES.index(label)))
                else:
                    print(f"[!] Warning: Unknown class '{label}' for file '{fname}'")
            else:
                missing_count += 1
                
        if missing_count > 0:
            print(f"[!] Warning: {missing_count} images from CSV not found on disk.")
        print(f"[*] Loaded {len(self.valid_data)} valid samples.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        path, label_idx = self.valid_data[idx]
        
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"[!] Error loading {path}: {e}")
            # Return a black image in case of error to prevent crash, or handle differently
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx

# ---------------- EVALUATION LOOP ----------------
def evaluate_metrics(data_dir, labels_csv, checkpoint_path, device='cpu', output_dir='eval_results'):
    print(f"[*] Starting Evaluation")
    print(f"    Data Dir: {data_dir}")
    print(f"    Labels CSV: {labels_csv}")
    print(f"    Checkpoint: {checkpoint_path}")
    
    # 1. Prepare Data
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = FlatImageDataset(data_dir, labels_csv, transform=eval_transform)
    if len(dataset) == 0:
        print("[!] Error: No valid data found. Exiting.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Load Model
    print(f"[*] Loading model: {MODEL_NAME}")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(state_dict, strict=True)
            print("[+] Weights loaded successfully.")
        except RuntimeError as e:
            print(f"[!] Error loading weights (Validation might be poor): {e}")
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
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Calculate Metrics
    print("\n[*] Evaluation Results")
    print("=" * 60)
    
    # Overall Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"[+] Overall Accuracy: {acc*100:.2f}%")
    print("-" * 60)

    # Detailed Classification Report
    print("[*] Detailed Classification Report:")
    report_dict = classification_report(all_labels, all_preds, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    report_str = classification_report(all_labels, all_preds, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES, zero_division=0)
    print(report_str)
    
    # Extract Macro and Weighted F1
    macro_f1 = report_dict['macro avg']['f1-score']
    weighted_f1 = report_dict['weighted avg']['f1-score']
    print("-" * 60)
    print(f"[+] Macro F1 Score:    {macro_f1:.4f}")
    print(f"[+] Weighted F1 Score: {weighted_f1:.4f}")
    print("=" * 60)

    # 5. Save Visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Report to CSV
    # Convert report dict to DF, put support as int
    report_df = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(csv_path)
    print(f"[+] Detailed report saved to: {csv_path}")
    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
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

    return acc, report_dict

# ---------------- MAIN ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ViT DIBaS Model (Flat Folder + CSV)')
    parser.add_argument('--data_dir', required=True, help='Path to folder containing images')
    parser.add_argument('--labels_csv', required=True, help='Path to CSV file with filenames and labels')
    parser.add_argument('--checkpoint', default='vit_dibas_v2.pth', help='Path to model checkpoint')
    parser.add_argument('--device', default='auto', help='Device to use (cpu/cuda)')
    parser.add_argument('--output_dir', default='eval_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        
    evaluate_metrics(args.data_dir, args.labels_csv, args.checkpoint, device, args.output_dir)
