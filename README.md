# 🩺 Skin Cancer Classification on ISIC-2017  
### Deep Learning Framework for Automated Melanoma Detection and Analysis  

---

## 📖 Abstract

Skin cancer is among the most common and deadly cancers worldwide.  
Early detection is vital for survival, yet manual diagnosis of dermoscopic images is subjective and prone to error.  
This repository presents a **complete deep-learning framework** for **automatic skin lesion classification** on the **ISIC-2017 Challenge Dataset**, designed for **reproducible research**, **clinical interpretability**, and **open benchmarking**.

The pipeline integrates advanced preprocessing techniques — including **hair removal**, **contrast enhancement**, and **image normalization** — with multiple state-of-the-art CNN architectures:  
**ConvNeXt-Tiny**, **DenseNet-201**, **EfficientNet-B3**, **Inception-V3**, **VGG-16**, **VGG-19**, and **Xception**.  
Each model is trained and evaluated under identical conditions to provide a fair comparison of diagnostic accuracy.

---

## 🎯 Objectives

1. Build a modular, reproducible deep-learning framework for skin lesion analysis.  
2. Evaluate and compare the performance of various CNN architectures on ISIC-2017.  
3. Investigate the effects of preprocessing (hair removal and contrast enhancement).  
4. Provide a clean, extensible foundation for future research and clinical applications.

---

## 🧩 Repository Structure

├── CITATION.cff # Citation metadata
├── configs/ # Model configuration files (YAML)
│ ├── base.yaml
│ ├── convnext_tiny.yaml
│ ├── densenet201.yaml
│ ├── efficientnet_b3.yaml
│ ├── inception_v3.yaml
│ ├── vgg16.yaml
│ └── xception.yaml
├── data/ # Data preparation and preprocessing
│ ├── contrast_enhanced.py # Contrast enhancement
│ ├── download_data.py # ISIC-2017 dataset downloader
│ ├── hair_remove.py # Hair removal using morphological filters
│ ├── prepare_data.py # CSV generation and dataset organization
│ └── init.py
├── src/ # Core training and evaluation modules
│ ├── config.py # Global parameters and paths
│ ├── dataset.py # Data loading utilities
│ ├── model.py # CNN model builder
│ ├── train.py # Training loop and callbacks
│ ├── evalute.py # Evaluation and metrics reporting
│ ├── utils.py # Helper utilities
│ └── init.py
├── results/ # Output artifacts
│ ├── metrics/ # Evaluation results per model
│ └── models/ # Trained weights
├── LICENSE # License file
├── Makefile # Simple automation commands
├── pyproject.toml # Project metadata
├── requirements.txt # Python dependencies
└── README.md # Documentation

yaml
Copy code

---

## 🧠 Theoretical Background

Dermoscopic image analysis is a classic computer-vision challenge characterized by variability in lesion color, size, and texture.  
Deep convolutional neural networks (CNNs) can learn hierarchical visual patterns that enable automated melanoma detection.

Our system benchmarks several families of architectures:
- **VGG** (deep, simple, uniform convolutional blocks)  
- **Inception** (multi-scale feature extraction)  
- **DenseNet** (dense connectivity for gradient efficiency)  
- **EfficientNet** (compound scaling for optimal accuracy vs. parameters)  
- **ConvNeXt** (modernized CNN inspired by transformers)  

By harmonizing preprocessing and evaluation, the framework isolates model differences from data inconsistencies — a key requirement for medical AI reproducibility.

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/skin-cancer.git
cd skin-cancer
2️⃣ Create and Activate a Virtual Environment
bash
Copy code
python3 -m venv venv
source venv/bin/activate    # macOS / Linux
# or
venv\Scripts\activate       # Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Verify Installation
bash
Copy code
python --version
Make sure it matches the Python version specified in pyproject.toml.

📦 Dataset
This project uses the ISIC-2017 Challenge Dataset for skin lesion classification.
Download automatically:

bash
Copy code
python data/download_data.py
Or manually from the official source:
🔗 ISIC 2017 Challenge Dataset

After download, run:

bash
Copy code
python data/prepare_data.py
Preprocessing
Step	Script	Description
Hair Removal	data/hair_remove.py	Removes hair artifacts using morphological filtering and inpainting
Contrast Enhancement	data/contrast_enhanced.py	Enhances lesion visibility by stretching image contrast
Dataset Preparation	data/prepare_data.py	Splits data into train/validation/test CSVs and structures folders

🧠 Model Training
Each architecture is defined by its YAML file under configs/.

Example (train EfficientNet-B3):

bash
Copy code
python src/train.py --config configs/efficientnet_b3.yaml
The configuration defines:

yaml
Copy code
model_name: EfficientNetB3
img_size: [224, 224]
epochs: 50
batch_size: 32
optimizer: Adam
learning_rate: 1e-4
loss: categorical_crossentropy
metrics: [accuracy, precision, recall, f1_score]
augmentation: False
Training Features
Model checkpointing and best-epoch saving

Early stopping and learning-rate scheduling

Real-time metric logging (TensorBoard support)

Automatic saving to:

bash
Copy code
results/models/<ModelName>/
results/metrics/evaluation_results_<ModelName>/
🧪 Evaluation
Evaluate a trained model:

bash
Copy code
python src/evalute.py --model EfficientNetB3
Outputs include:

Classification report (precision, recall, F1)

Confusion matrix

ROC-AUC score

Accuracy plots and ROC curves
Results are saved to results/metrics/.

📊 Experimental Results
Illustrative benchmark (ISIC-2017, preprocessed data):

Model	Accuracy	Precision	Recall	F1-Score	Parameters (M)
VGG16	0.874	0.861	0.866	0.863	138.4
VGG19	0.878	0.865	0.870	0.867	143.7
Inception-V3	0.892	0.884	0.889	0.886	23.8
DenseNet-201	0.904	0.896	0.905	0.900	20.2
EfficientNet-B3	0.912	0.907	0.909	0.908	12.0
Xception	0.907	0.899	0.905	0.902	22.9
ConvNeXt-Tiny	0.909	0.901	0.904	0.903	28.6

(Replace with your actual results once experiments complete.)

🧰 Utilities and Automation
Makefile Shortcuts
bash
Copy code
make setup         # install dependencies
make train CONFIG=configs/efficientnet_b3.yaml
make evaluate MODEL=EfficientNetB3
make clean         # remove temporary logs
TensorBoard Visualization
bash
Copy code
tensorboard --logdir results/logs/
Visualize:

Accuracy vs. epochs

Loss curves

ROC curves

Per-class confusion matrices

🧬 Research Impact
This repository promotes transparency and reproducibility in medical image analysis.
By providing a shared, well-documented baseline, it helps researchers:

Evaluate preprocessing methods systematically

Benchmark architectures fairly

Build explainable models for clinical support

🚀 Future Work
Integration of Vision Transformers (ViT, Swin-Transformer)

Addition of explainable AI modules (Grad-CAM, SHAP)

Multi-dataset experiments (ISIC-2020, HAM10000)

Domain adaptation and color-harmonization for multi-scanner data

Semi-supervised and federated learning extensions

🧾 License
This project is released under the MIT License.
See the LICENSE file for full terms.

🧑‍💻 Citation
If you use this code or ideas in your work, please cite:

csharp
Copy code
Mikaeelzadeh, K. (2025). Skin Cancer Classification on ISIC-2017: 
Deep Learning Pipeline for Melanoma Detection. 
GitHub Repository: https://github.com/kamyarmikaeelzade/skin-cancer
A formal citation record is included in the CITATION.cff file.

📬 Contact
Author: Kamyar Mikaeelzade
Field: Biomedical Engineering – Artificial Intelligence in Medical Imaging
Email: kamyarmikaeelzadeh@gmail.com
GitHub: github.com/kamyarmikaeelzade
LinkedIn: linkedin.com/in/kamyarmikaeelzade

🌍 Acknowledgments
ISIC Archive for open-access medical imaging datasets

TensorFlow and Keras open-source communities

Amirkabir University of Technology (Tehran Polytechnic) for academic guidance

Fellow researchers contributing to ethical AI in healthcare

🧭 Reproducibility Checklist
✅ Dataset link and preprocessing scripts included
✅ Model configurations version-controlled in YAML
✅ Random seeds fixed for deterministic results
✅ Training logs and metrics automatically saved
✅ Evaluation and visualization tools provided
✅ All dependencies declared in requirements.txt and pyproject.toml

🧠 Conclusion
This repository provides a reproducible, extensible, and research-grade pipeline for melanoma classification from dermoscopic images.
Through systematic preprocessing, standardized training, and open comparison of architectures, it establishes a strong foundation for future work in AI-based dermatology and medical imaging.

“Reproducibility is not repetition — it is the language through which science speaks truth.”

pgsql
Copy code

---

This version is ready to publish — it’s structured for both **academic readers** (clear methodology, citation, impact) and **GitHub visitors** (installation, usage, results, contact).  

Would you like me to add **visual badges** (e.g., Python 3.10, TensorFlow 2.x, MIT License, Stars, Issues) and **a project banner image section** to make it look like a top-tier GitHub project page?




