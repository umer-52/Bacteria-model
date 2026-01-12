"""
GramViT Batch Inference Script
Process multiple images and generate a summary report.
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
from pathlib import Path
import glob

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
    print(f"Loading model: {model_name}")
    
    # Create model
    model = create_model(model_name, pretrained=False, drop_path_rate=0.0)
    model.to(device)
    model.eval()
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
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
        print("Checkpoint loaded successfully\n")
    else:
        print("WARNING: No checkpoint provided or file not found. Using randomly initialized model.\n")
    
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


def predict(model, image_tensor, device='cpu', num_samples=1):
    """Run inference on image."""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
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


def process_image(model, image_path, device, img_size, num_samples):
    """Process a single image and return results."""
    try:
        image_tensor = preprocess_image(image_path, img_size)
        probs, pred_class, pred_name = predict(model, image_tensor, device, num_samples)
        
        return {
            'image': os.path.basename(image_path),
            'path': image_path,
            'predicted_class': {
                'index': int(pred_class),
                'name': pred_name,
                'confidence': float(probs[pred_class])
            },
            'probabilities': {
                CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probs)
            },
            'status': 'success'
        }
    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else repr(e)
        error_traceback = traceback.format_exc()
        return {
            'image': os.path.basename(image_path),
            'path': image_path,
            'status': 'error',
            'error': error_msg,
            'traceback': error_traceback
        }


def main():
    parser = argparse.ArgumentParser(description='GramViT Batch Image Inference')
    parser.add_argument('--image_dir', required=True, type=str,
                       help='Directory containing images to process')
    parser.add_argument('--checkpoint', required=True, type=str,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--model', default='longvit_small_patch32_4096_gs_classification',
                       type=str, help='Model name')
    parser.add_argument('--img_size', default=4096, type=int,
                       help='Input image size (4096 or 1024)')
    parser.add_argument('--device', default='auto', type=str,
                       help='Device to use: auto, cpu, or cuda')
    parser.add_argument('--num_samples', default=1, type=int,
                       help='Number of inference samples to average')
    parser.add_argument('--output_file', default=None, type=str,
                       help='Path to save results JSON file')
    parser.add_argument('--extensions', default='jpg,jpeg,png,bmp,tiff', type=str,
                       help='Comma-separated image extensions to process')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Get image files
    extensions = [ext.strip() for ext in args.extensions.split(',')]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, f'*.{ext}')))
        image_files.extend(glob.glob(os.path.join(args.image_dir, f'*.{ext.upper()}')))
    
    if not image_files:
        print(f"ERROR: No images found in {args.image_dir}")
        return
    
    print("GramViT Batch Inference")
    print("=" * 70)
    print(f"Image directory: {args.image_dir}")
    print(f"Found {len(image_files)} images")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Image size: {args.img_size}")
    print("=" * 70)
    
    # Load model
    model = load_model(args.checkpoint, args.model, args.img_size, device)
    
    # Process all images
    results = []
    print(f"\nProcessing {len(image_files)} images...\n")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}", end=' ... ')
        result = process_image(model, image_path, device, args.img_size, args.num_samples)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"OK - {result['predicted_class']['name']} ({result['predicted_class']['confidence']*100:.1f}%)")
        else:
            error_msg = result.get('error', 'Unknown error')
            if not error_msg:
                error_msg = result.get('traceback', 'Unknown error (no error message)')
            print(f"ERROR: {error_msg[:200]}")  # Limit to first 200 chars
    
    # Generate summary
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print(f"\nSuccessfully processed: {len(successful)}/{len(results)}")
    if failed:
        print(f"Failed: {len(failed)}")
    
    # Class distribution
    if successful:
        print(f"\nClass Distribution:")
        class_counts = {}
        for r in successful:
            class_name = r['predicted_class']['name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(successful)) * 100
            print(f"  {class_name:50s} {count:3d} ({percentage:5.1f}%)")
    
    # Detailed results
    print(f"\nDetailed Results:")
    print("-" * 70)
    for r in successful:
        print(f"\n  Image: {r['image']}")
        print(f"  Prediction: {r['predicted_class']['name']}")
        print(f"  Confidence: {r['predicted_class']['confidence']*100:.2f}%")
        print(f"  Top probabilities:")
        sorted_probs = sorted(r['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs[:3]:
            print(f"    - {class_name:45s} {prob*100:5.2f}%")
    
    if failed:
        print(f"\nFailed Images:")
        for r in failed:
            print(f"  - {r['image']}: {r.get('error', 'Unknown error')}")
    
    # Save results
    output_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_dir': args.image_dir,
        'checkpoint': args.checkpoint,
        'model': args.model,
        'image_size': args.img_size,
        'total_images': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'class_distribution': class_counts if successful else {},
        'results': results
    }
    
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(args.image_dir, f"batch_inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)
    print("Batch inference complete!")
    print("=" * 70)
    
    return output_data


if __name__ == '__main__':
    main()
