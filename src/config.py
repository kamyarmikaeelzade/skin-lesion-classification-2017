# src/config.py
"""
Global configuration for ISIC2017 Melanoma CNN Benchmark.
Adjust paths and hyperparameters here.
"""

# ===== Paths (relative to repo root) =====
TRAIN_CSV = "data/ISIC-2017_Training_Part3_GroundTruth.csv"
VALID_CSV = "data/ISIC-2017_Validation_Part3_GroundTruth.csv"
TEST_CSV  = "data/ISIC-2017_Test_v2_Part3_GroundTruth.csv"

TRAIN_DIR = "data/ISIC-2017_Training_Data_contrast"
VALID_DIR = "data/ISIC-2017_Validation_Data_contrast"
TEST_DIR  = "data/ISIC-2017_Test_v2_Data_contrast"

EXPERIMENT_DIR = "experiments"
LOG_DIR = f"{EXPERIMENT_DIR}/logs"
RESULTS_DIR = f"{EXPERIMENT_DIR}/results"
CHECKPOINT_PATH = f"{RESULTS_DIR}/best_model.keras"
HISTORY_CSV = f"{RESULTS_DIR}/train_history.csv"
PLOT_ROC_PATH = f"{RESULTS_DIR}/roc_curve.png"
PLOT_CM_PATH = f"{RESULTS_DIR}/confusion_matrix.png"

# ===== Training =====
IMG_SIZE = (224, 224)      # (H, W)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0         # If you switch to AdamW
MIXED_PRECISION = False    # Set True on GPUs with AMP support

# ===== Model =====
BACKBONE = "EfficientNetB0"    # Options in models.list_backbones()
FREEZE_BACKBONE = True
DROPOUT = 0.3

# ===== Task =====
NUM_CLASSES = 1            # Binary
CLASS_MODE = "binary"

# ===== Repro =====
SEED = 42
