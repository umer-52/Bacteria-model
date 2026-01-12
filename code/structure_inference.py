"""
Structural Inference Script
Predicts Gram Stain, Morphology, and Arrangement directly from the ViT model.
Supports Test Time Augmentation (TTA) for improved stability.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

# ---------------- CONFIGURATION ----------------
MODEL_ARCH = 'vit_base_patch16_224'
NUM_CLASSES = 33

# ---------------- BIOLOGICAL MAPPING ----------------
# Maps model's class index (0-32) to physical traits
TRAIT_MAP = {
    0:  {"gram": "Negative", "morph": "Rod",   "arr": "Scattered"},       # Acinetobacter.baumanii
    1:  {"gram": "Positive", "morph": "Rod",   "arr": "Filamentous"},     # Actinomyces.israeli
    2:  {"gram": "Negative", "morph": "Rod",   "arr": "Scattered"},       # Bacteroides.fragilis
    3:  {"gram": "Positive", "morph": "Rod",   "arr": "Branched"},        # Bifidobacterium.spp
    4:  {"gram": "Yeast",    "morph": "Round", "arr": "Clusters"},        # Candida.albicans
    5:  {"gram": "Positive", "morph": "Rod",   "arr": "Scattered"},       # Clostridium.perfringens
    6:  {"gram": "Positive", "morph": "Cocci", "arr": "Pairs/Chains"},    # Enterococcus.faecalis
    7:  {"gram": "Positive", "morph": "Cocci", "arr": "Pairs/Chains"},    # Enterococcus.faecium
    8:  {"gram": "Negative", "morph": "Rod",   "arr": "Scattered"},       # Escherichia.coli
    9:  {"gram": "Negative", "morph": "Rod",   "arr": "Fusiform"},        # Fusobacterium.spp
    10: {"gram": "Positive", "morph": "Rod",   "arr": "Chains"},          # Lactobacillus.casei
    11: {"gram": "Positive", "morph": "Rod",   "arr": "Chains"},          # Lactobacillus.crispatus
    12: {"gram": "Positive", "morph": "Rod",   "arr": "Chains"},          # Lactobacillus.delbrueckii
    13: {"gram": "Positive", "morph": "Rod",   "arr": "Chains"},          # Lactobacillus.gasseri
    14: {"gram": "Positive", "morph": "Rod",   "arr": "Chains"},          # Lactobacillus.jehnsenii
    15: {"gram": "Positive", "morph": "Rod",   "arr": "Chains"},          # Lactobacillus.johnsonii
    16: {"gram": "Positive", "morph": "Rod",   "arr": "Chains"},          # Lactobacillus.paracasei
    17: {"gram": "Positive", "morph": "Rod",   "arr": "Chains"},          # Lactobacillus.plantarum
    18: {"gram": "Positive", "morph": "Rod",   "arr": "Chains"},          # Lactobacillus.reuteri
    19: {"gram": "Positive", "morph": "Rod",   "arr": "Chains"},          # Lactobacillus.rhamnosus
    20: {"gram": "Positive", "morph": "Rod",   "arr": "Chains"},          # Lactobacillus.salivarius
    21: {"gram": "Positive", "morph": "Rod",   "arr": "Scattered"},       # Listeria.monocytogenes
    22: {"gram": "Positive", "morph": "Cocci", "arr": "Tetrads/Clusters"},# Micrococcus.spp
    23: {"gram": "Negative", "morph": "Cocci", "arr": "Pairs"},           # Neisseria.gonorrhoeae
    24: {"gram": "Negative", "morph": "Rod",   "arr": "Scattered"},       # Porphyromonas.gingivalis
    25: {"gram": "Positive", "morph": "Rod",   "arr": "Clusters"},        # Propionibacterium.acnes
    26: {"gram": "Negative", "morph": "Rod",   "arr": "Scattered"},       # Proteus.spp
    27: {"gram": "Negative", "morph": "Rod",   "arr": "Scattered"},       # Pseudomonas.aeruginosa
    28: {"gram": "Positive", "morph": "Cocci", "arr": "Clusters"},        # Staphylococcus.aureus
    29: {"gram": "Positive", "morph": "Cocci", "arr": "Clusters"},        # Staphylococcus.epidermidis
    30: {"gram": "Positive", "morph": "Cocci", "arr": "Clusters"},        # Staphylococcus.saprophyticus
    31: {"gram": "Positive", "morph": "Cocci", "arr": "Chains"},          # Streptococcus.agalactiae
    32: {"gram": "Negative", "morph": "Cocci", "arr": "Clusters"}         # Veillonella.spp
}


def load_model(checkpoint_path, device='cpu'):
    print("[*] Loading model...")
    model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=NUM_CLASSES)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        print("[+] Checkpoint loaded successfully")
    else:
        print(f"[!] Warning: Checkpoint not found at {checkpoint_path}")
    
    model.to(device)
    model.eval()
    return model


def aggregate_to_traits(probs):
    """Aggregate class probabilities into trait scores."""
    gram_scores = {}
    morph_scores = {}
    arr_scores = {}
    
    for idx, score in enumerate(probs):
        if idx not in TRAIT_MAP:
            continue
        traits = TRAIT_MAP[idx]
        gram_scores[traits['gram']] = gram_scores.get(traits['gram'], 0.0) + score
        morph_scores[traits['morph']] = morph_scores.get(traits['morph'], 0.0) + score
        arr_scores[traits['arr']] = arr_scores.get(traits['arr'], 0.0) + score
    
    return {
        "Gram Stain":  max(gram_scores.items(), key=lambda x: x[1]),
        "Morphology":  max(morph_scores.items(), key=lambda x: x[1]),
        "Arrangement": max(arr_scores.items(), key=lambda x: x[1])
    }


def predict_structure(model, image_path, device='cpu'):
    """Standard single-pass prediction."""
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    return aggregate_to_traits(probs)


def predict_with_tta(model, image_path, device='cpu'):
    """
    Test Time Augmentation (TTA) prediction.
    Averages predictions across 4 views: original, H-flip, V-flip, 90-degree rotation.
    """
    img = Image.open(image_path).convert('RGB')
    
    # Slightly larger resize + center crop to avoid edge artifacts
    base_transform = transforms.Compose([
        transforms.Resize(248),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create 4 variations for TTA
    inputs = []
    
    # View 1: Standard
    inputs.append(base_transform(img))
    
    # View 2: Horizontal Flip
    inputs.append(base_transform(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    
    # View 3: Vertical Flip
    inputs.append(base_transform(img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)))
    
    # View 4: Rotate 90 degrees
    inputs.append(base_transform(img.rotate(90)))
    
    # Stack into batch [4, 3, 224, 224]
    batch_tensor = torch.stack(inputs).to(device)
    
    print("[*] Aggregating 4 views (TTA) for improved stability...")
    with torch.no_grad():
        logits = model(batch_tensor)
        # Average probabilities across all 4 views
        avg_probs = F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
    
    return aggregate_to_traits(avg_probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Structural Inference - Gram/Morphology/Arrangement')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--checkpoint', default='vit_dibas_best.pth', help='Path to .pth file')
    parser.add_argument('--device', default='auto', help='Device: auto, cpu, cuda')
    parser.add_argument('--tta', action='store_true', help='Enable Test Time Augmentation (slower but more accurate)')
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    model = load_model(args.checkpoint, device)
    
    print(f"\n[*] Analyzing Image Structure: {os.path.basename(args.image)}")
    if args.tta:
        print("[*] Mode: TTA (4-view ensemble)")
    else:
        print("[*] Mode: Standard (single-pass)")
    print("-" * 50)
    
    # Choose prediction method
    if args.tta:
        results = predict_with_tta(model, args.image, device)
    else:
        results = predict_structure(model, args.image, device)
    
    for trait, (val, conf) in results.items():
        bar_len = int(conf * 20) 
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"{trait:12s} : {val:15s}  {conf*100:5.1f}%  [{bar}]")
    
    print("-" * 50)
    print("[+] Structure analysis complete!")