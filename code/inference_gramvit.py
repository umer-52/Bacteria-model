"""
GramViT Single Image Inference Script
Run inference on a single image to get Gram-stain classification predictions.
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from timm import create_model
from datetime import datetime

# Add code directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modeling_finetune  # noqa: F401 - registers models
import utils


# Class names for the 5 categories
CLASS_NAMES = [
    "GPC-clusters (Gram-positive cocci in clusters)",
    "GPC-pairschains (Gram-positive cocci in pairs/chains)",
    "GNR (Gram-negative rods)",
    "GPR (Gram-positive rods)",
    "No bacteria"
]


def load_model(checkpoint_path, model_name='longvit_small_patch32_4096_gs_classification', 
               img_size=4096, device='cpu'):
    """Load the trained GramViT model from checkpoint."""
    print(f"📦 Loading model: {model_name}")
    
    # Create model
    model = create_model(model_name, pretrained=False, drop_path_rate=0.0)
    model.to(device)
    model.eval()
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        print("✅ Checkpoint loaded successfully")
    else:
        print("⚠️  No checkpoint provided or file not found. Using randomly initialized model.")
    
    return model


def preprocess_image(image_path, img_size=4096):
    """Preprocess image for model input."""
    # Load image
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('RGB')
    else:
        img = image_path
    
    # Resize to model input size
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor


def predict(model, image_tensor, device='cpu', num_samples=4):
    """
    Run inference on image.
    
    Args:
        model: Trained GramViT model
        image_tensor: Preprocessed image tensor [1, 3, H, W]
        device: Device to run on
        num_samples: Number of times to sample (for consistency with training)
    
    Returns:
        probs: Class probabilities [5]
        pred_class: Predicted class index
        pred_name: Predicted class name
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # During training, they average over 4 samples, but for single image inference
        # we can just run it once or multiple times with the same image
        logits_list = []
        
        for _ in range(num_samples):
            logits = model(image=image_tensor)
            logits_list.append(logits)
        
        # Average logits if multiple samples
        if num_samples > 1:
            logits = torch.mean(torch.stack(logits_list), dim=0)
        else:
            logits = logits_list[0]
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_name = CLASS_NAMES[pred_class]
    
    return probs[0].cpu(), pred_class, pred_name


def main():
    parser = argparse.ArgumentParser(description='GramViT Single Image Inference')
    parser.add_argument('--image', required=True, type=str, 
                       help='Path to input image file')
    parser.add_argument('--checkpoint', required=True, type=str,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--model', default='longvit_small_patch32_4096_gs_classification', 
                       type=str, help='Model name')
    parser.add_argument('--img_size', default=4096, type=int,
                       help='Input image size (4096 or 1024)')
    parser.add_argument('--device', default='auto', type=str,
                       help='Device to use: auto, cpu, or cuda')
    parser.add_argument('--num_samples', default=1, type=int,
                       help='Number of inference samples to average (default: 1)')
    parser.add_argument('--output_file', default=None, type=str,
                       help='Path to save results JSON file (optional)')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save results to a JSON file automatically')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("🔬 GramViT Inference")
    print("=" * 60)
    print(f"📷 Image: {args.image}")
    print(f"💾 Checkpoint: {args.checkpoint}")
    print(f"🖥️  Device: {device}")
    print(f"📐 Image size: {args.img_size}")
    print("=" * 60)
    
    # Load model
    model = load_model(args.checkpoint, args.model, args.img_size, device)
    
    # Preprocess image
    print(f"\n🔄 Preprocessing image...")
    try:
        image_tensor = preprocess_image(args.image, args.img_size)
        print(f"✅ Image loaded and preprocessed: {image_tensor.shape}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Run inference
    print(f"\n🧠 Running inference...")
    probs, pred_class, pred_name = predict(model, image_tensor, device, args.num_samples)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 PREDICTION RESULTS")
    print("=" * 60)
    print(f"\n🎯 Predicted Class: {pred_class} - {pred_name}")
    print(f"\n📈 Class Probabilities:")
    for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        bar_length = int(prob * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {i}. {class_name:45s} {prob*100:5.2f}% [{bar}]")
    
    # Prepare results dictionary
    results = {
        'image_path': args.image,
        'checkpoint': args.checkpoint,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'predicted_class': {
            'index': int(pred_class),
            'name': pred_name,
            'confidence': float(probs[pred_class])
        },
        'all_probabilities': {
            CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probs)
        },
        'model': args.model,
        'image_size': args.img_size
    }
    
    # Save results to file
    if args.save_results or args.output_file:
        if args.output_file:
            output_path = args.output_file
        else:
            # Auto-generate filename based on input image
            base_name = os.path.splitext(os.path.basename(args.image))[0]
            output_dir = os.path.dirname(args.image) if os.path.dirname(args.image) else '.'
            output_path = os.path.join(output_dir, f"{base_name}_gramvit_results.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ Inference complete!")
    print("=" * 60)
    
    # Return results for programmatic access
    return results


if __name__ == '__main__':
    main()

