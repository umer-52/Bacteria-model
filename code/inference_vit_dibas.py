"""
ViT DIBaS Inference Script (GramViT Style)
Run inference on images using the vit_dibas_best.pth model.
Produces hierarchical predictions: gram_stain, morphology, arrangement.
"""

import os
import sys
import argparse
import json
import time
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
import timm
from datetime import datetime

# ---------------- CONFIGURATION ----------------
DEFAULT_CSV_PATH = "Bacteria_Dataset/labeled_dataset.csv" 
MODEL_ARCH = 'vit_base_patch16_224'
IMG_SIZE = 224
NUM_CLASSES = 33

# ---------------- CLASS METADATA ----------------
# Maps class index to species info (gram_status, morphology, arrangement)
# This will be populated from the CSV or use defaults
CLASS_METADATA = {}

# Known gram stain categories
GRAM_CATEGORIES = ["Gram-positive", "Gram-negative"]

# Known morphology categories  
MORPHOLOGY_CATEGORIES = ["Cocci", "Rods", "Spirilla", "Bacilli", "Coccobacilli"]

# Known arrangement categories
ARRANGEMENT_CATEGORIES = ["Clusters", "Chains", "Pairs", "Scattered", "Filaments", "Singles"]


def load_class_metadata(csv_path):
    """Load class metadata from CSV to map species to gram/morphology/arrangement."""
    metadata = {}
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            # Build mapping from species to metadata
            for _, row in df.iterrows():
                species = row.get('species', '')
                if species and species not in metadata:
                    metadata[species] = {
                        'gram_status': row.get('gram_status', 'Unknown'),
                        'morphology': row.get('morphology', 'Unknown'),
                        'arrangement': row.get('arrangement', 'Unknown')
                    }
            
            # Create class index to species mapping (alphabetically sorted)
            species_list = sorted(metadata.keys())
            for idx, species in enumerate(species_list):
                CLASS_METADATA[idx] = {
                    'species': species,
                    **metadata[species]
                }
                
            print(f"✅ Loaded metadata for {len(CLASS_METADATA)} species from CSV")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not parse CSV: {e}")
    
    return CLASS_METADATA


def get_class_names_from_metadata():
    """Get ordered list of species names."""
    if CLASS_METADATA:
        return [CLASS_METADATA[i]['species'] for i in range(len(CLASS_METADATA))]
    return [f"Class {i}" for i in range(NUM_CLASSES)]


def load_model(checkpoint_path, device='cpu'):
    """Load the trained ViT model from checkpoint."""
    print(f"📦 Loading model: {MODEL_ARCH}")
    
    model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=NUM_CLASSES)
    model.to(device)
    model.eval()
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
            
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        print("✅ Checkpoint loaded successfully")
    else:
        print("⚠️  Warning: Checkpoint not found! Using random weights.")
    
    return model


def preprocess_image(image_path, img_size=224):
    """Preprocess image for model input."""
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('RGB')
    else:
        img = image_path
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(img).unsqueeze(0)


def aggregate_predictions(probs, category_type):
    """
    Aggregate class probabilities by category (gram_stain, morphology, or arrangement).
    Returns the most likely category and its aggregated confidence.
    """
    category_probs = {}
    
    for idx, prob in enumerate(probs.tolist()):
        if idx in CLASS_METADATA:
            meta = CLASS_METADATA[idx]
            
            if category_type == 'gram_stain':
                # Map to Gram-positive or Gram-negative
                gram = meta.get('gram_status', 'Unknown')
                if gram == 'Positive':
                    cat = 'Gram-positive'
                elif gram == 'Negative':
                    cat = 'Gram-negative'
                else:
                    cat = gram
            elif category_type == 'morphology':
                morph = meta.get('morphology', 'Unknown')
                # Simplify morphology labels
                if 'Cocci' in morph:
                    cat = 'Cocci'
                elif 'Bacilli' in morph or 'Rods' in morph:
                    cat = 'Rods'
                elif 'Spirilla' in morph or 'Spiral' in morph:
                    cat = 'Spirilla'
                else:
                    cat = morph
            elif category_type == 'arrangement':
                cat = meta.get('arrangement', 'Unknown')
            else:
                cat = 'Unknown'
            
            category_probs[cat] = category_probs.get(cat, 0.0) + prob
    
    if not category_probs:
        return 'Unknown', 0.0
    
    # Find the category with highest aggregated probability
    best_cat = max(category_probs, key=category_probs.get)
    best_conf = category_probs[best_cat]
    
    return best_cat, best_conf


def predict(model, image_tensor, device='cpu'):
    """Run inference on image."""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
    return probs[0].cpu(), pred_idx


def main():
    parser = argparse.ArgumentParser(description='ViT DIBaS Inference')
    parser.add_argument('--image', required=True, type=str, help='Path to input image')
    parser.add_argument('--checkpoint', default='vit_dibas_best.pth', type=str, help='Path to .pth file')
    parser.add_argument('--labels_csv', default=DEFAULT_CSV_PATH, type=str, help='Path to csv with class mappings')
    parser.add_argument('--device', default='auto', type=str, help='Device: auto, cpu, cuda')
    parser.add_argument('--output_file', default=None, type=str, help='Custom output JSON path')
    parser.add_argument('--no_save', action='store_true', help='Skip saving JSON results')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Load class metadata
    load_class_metadata(args.labels_csv)
    class_names = get_class_names_from_metadata()

    print("🔬 ViT DIBaS Inference")
    print("=" * 60)
    print(f"📷 Image: {args.image}")
    print(f"💾 Checkpoint: {args.checkpoint}")
    print(f"🖥️  Device: {device}")
    print(f"📐 Image size: {IMG_SIZE}")
    print(f"🏷️  Species loaded: {len(CLASS_METADATA)}")
    print("=" * 60)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    print(f"\n🔄 Preprocessing image...")
    try:
        image_tensor = preprocess_image(args.image, IMG_SIZE)
        print(f"✅ Image loaded: {image_tensor.shape}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

    print(f"\n🧠 Running inference...")
    
    # Time the inference
    start_time = time.time()
    probs, pred_idx = predict(model, image_tensor, device)
    inference_time = time.time() - start_time
    
    # Get species prediction
    if pred_idx < len(class_names):
        pred_species = class_names[pred_idx]
    else:
        pred_species = f"Class {pred_idx}"
    
    # Aggregate predictions by category
    gram_label, gram_conf = aggregate_predictions(probs, 'gram_stain')
    morph_label, morph_conf = aggregate_predictions(probs, 'morphology')
    arr_label, arr_conf = aggregate_predictions(probs, 'arrangement')
    
    # Build overall prediction string
    overall_prediction = f"{gram_label} {morph_label.lower()} in {arr_label.lower()}"

    # --- DISPLAY RESULTS ---
    print("\n" + "=" * 60)
    print("📊 PREDICTION RESULTS")
    print("=" * 60)
    print(f"\n🎯 Species: {pred_species} ({probs[pred_idx]*100:.1f}%)")
    print(f"\n📋 Hierarchical Predictions:")
    print(f"   • Gram Stain:  {gram_label} ({gram_conf*100:.1f}%)")
    print(f"   • Morphology:  {morph_label} ({morph_conf*100:.1f}%)")
    print(f"   • Arrangement: {arr_label} ({arr_conf*100:.1f}%)")
    print(f"\n🏷️  Overall: {overall_prediction}")
    print(f"⏱️  Inference time: {inference_time:.2f}s")

    # --- BUILD RESULTS JSON (Hierarchical Format) ---
    results = {
        "image_id": os.path.basename(args.image),
        "predictions": {
            "gram_stain": {
                "label": gram_label,
                "confidence": round(gram_conf, 3)
            },
            "morphology": {
                "label": morph_label,
                "confidence": round(morph_conf, 3)
            },
            "arrangement": {
                "label": arr_label,
                "confidence": round(arr_conf, 3)
            }
        },
        "overall_prediction": overall_prediction,
        "inference_time": f"{inference_time:.2f}s",
        "species_prediction": {
            "name": pred_species,
            "confidence": round(float(probs[pred_idx]), 3)
        },
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if not args.no_save:
        if args.output_file:
            output_path = args.output_file
        else:
            base_name = os.path.splitext(os.path.basename(args.image))[0]
            output_dir = os.path.dirname(args.image) if os.path.dirname(args.image) else '.'
            output_path = os.path.join(output_dir, f"{base_name}_vit_results.json")
            
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")

    print("\n" + "=" * 60)
    print("✅ Inference complete!")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
