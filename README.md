# Bacteria Classification Model (ViT DIBaS)

This project implements a **Vision Transformer (ViT)** model for classifying bacterial colonies (33 species) using the DIBaS dataset. It achieves high accuracy (99%+) on structural traits and species prediction.

## 🧠 Model Details

- **Architecture**: `vit_base_patch16_224` (Vision Transformer, Base, 16x16 patches, 224x224 input)
- **Library**: `timm` (PyTorch Image Models)
- **Classes**: 33 Bacterial Species
- **Pretrained Weights**: `vit_dibas_v2.pth` (Best performing checkpoint, ~99-100% confidence on test samples)

## 📂 Repository Structure

```
GramViT-master/
├── code/
│   ├── train_vit_dibas.py      # Training script (Kaggle compatible)
│   ├── evaluate_vit_dibas.py   # Evaluation script (Metrics & Plots)
│   └── (Original GramViT scripts...)
├── Bacteria_Dataset/           # Dataset folder (Git ignored)
├── vit_dibas_v2.pth            # Trained Model Weights (Git ignored)
└── README.md
```

## 🚀 Getting Started

### 1. Prerequisites
Install the required Python packages:

```bash
pip install torch torchvision timm pandas seaborn matplotlib scikit-learn
```

### 2. Training (Fine-tuning)

To train the model on your own dataset (or DIBaS), use the provided training script. It is configured to run smoothly on Kaggle or local GPUs.

```bash
# Run the training script (Modify paths inside the file if needed)
python code/train_vit_dibas.py
```

**Features:**
- Automatically merges new data with the original DIBaS dataset.
- Cleans corrupt images before training.
- Saves the best model checkpoint based on validation accuracy.

### 3. Evaluation

Evaluate the trained model on a test set to get detailed metrics (Accuracy, Precision, Recall, F1-score) and a confusion matrix.

```bash
python code/evaluate_vit_dibas.py --data_dir "path/to/test_dataset" --checkpoint vit_dibas_v2.pth
```

**Outputs:**
- `eval_results/classification_report.csv`: Detailed metrics for all 33 classes.
- `eval_results/confusion_matrix.png`: Heatmap of predictions.

---

## 🦠 Legacy: GramViT (Original Project)

*The following section refers to the original GramViT project for whole-slide images.*

The repo contains the trained model and code for the paper, [A Novel Framework for the Automated Characterization of Gram Stained Blood Culture Slides Using a Large Scale Vision Transformer](https://pages.github.com/). GramViT is a region-sampling framework for characterizing Gram-stained slides using Microsoft's LongViT vision Transformer model.

**Original scripts:** `inference_gramvit.py`, `batch_inference_gramvit.py`, `launch_finetuning_gs.sh`.
