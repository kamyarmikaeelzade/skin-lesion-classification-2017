

# Skin Lesion Classification – ISIC 2017

## 📌 Overview
This repository contains a deep learning pipeline for **skin lesion classification** using the [ISIC 2017 Challenge Dataset](https://challenge.isicarchive.com).  
The goal is to automatically classify dermoscopic images into three categories:
- **Melanoma**
- **Seborrheic Keratosis**
- **Benign Nevi**

The project addresses common challenges in medical imaging:
- **Class imbalance** – Melanoma is under-represented compared to benign classes.
- **High intra-class variability** – Significant differences in lesion appearance within the same category.
- **Inter-class similarity** – Visual overlap between melanoma and benign nevi.
- **Image artifacts** – Hair, lighting variations, and color inconsistencies.

---

## 📂 Dataset
- **Source:** International Skin Imaging Collaboration (ISIC) 2017  
- **Training samples:** 2000+ dermoscopic images  
- **Image format:** JPEG, RGB  
- **Ground truth:** Provided by dermatology experts  
- **Pre-split:** Train / Validation / Test sets

**Data Preprocessing:**
1. Resize images to fixed dimensions (default: 224×224 or 384×384)
2. Hair removal preprocessing (optional)
3. Color normalization (Shades of Gray / Retinex)
4. Data augmentation (horizontal/vertical flips, rotation, scaling, MixUp, CutMix)

---

## 🧠 Model Architectures
Supported architectures:
- **ResNet-50 / ResNet-152**
- **DenseNet-169**
- **EfficientNet-B4**
- **Vision Transformer (ViT)**
- **ConvNeXt-T/B**

Custom loss functions:
- **Focal Loss**
- **Class-balanced Loss**

---

## 📊 Evaluation Metrics
- Accuracy
- Sensitivity (Recall)
- Specificity
- Balanced Accuracy
- AUC (per class & macro-average)
- Matthews Correlation Coefficient (MCC)

All metrics follow medical AI reporting guidelines (*Nature Medicine*, 2021).

---

## ⚙️ Installation
```bash
git clone https://github.com/kamyarmikaeelzade/skin-lesion-classification-2017.git
cd skin-lesion-classification-2017
pip install -r requirements.txt
```

---

## 🚀 Usage
### Train the model
```bash
python train.py --model efficientnet_b4 --epochs 50 --batch-size 32 --lr 1e-4
```

### Evaluate the model
```bash
python evaluate.py --model efficientnet_b4 --weights best_model.pth
```

**Options:**
- `--model`: Model architecture (`resnet50`, `convnext_tiny`, `vit_base`, etc.)
- `--loss`: Loss function (`focal`, `ce`, `cb_loss`)
- `--hair-removal`: Enable hair removal preprocessing
- `--color-correction`: Apply color constancy normalization

---

## 🖥 HPC & CI/CD
- Dockerfile with NVIDIA GPU support
- GitHub Actions workflow for automated testing
- DVC for dataset and model version tracking

---

## 📚 References
1. Codella et al., "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the ISIC 2017", IEEE ISBI, 2018.
2. Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML, 2019.
3. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition", ICLR, 2021.

---

## 📝 License
MIT License.

---

## ✨ Citation
```bibtex
@article{skinlesion2017,
  author={Mikaeelzade, Kamyar},
  title={Skin Lesion Classification Using Deep Learning and ISIC 2017 Dataset},
  journal={GitHub Repository},
  year={2025},
  url={https://github.com/kamyarmikaeelzade/skin-lesion-classification-2017}