# src/evaluate.py
"""
Evaluation:
- loads best checkpoint
- computes ROC AUC on val/test
- saves ROC and Confusion Matrix images
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

from .dataset import create_data_generators
from .config import CHECKPOINT_PATH, PLOT_ROC_PATH, PLOT_CM_PATH, RESULTS_DIR
from .utils import get_logger, ensure_dirs

def _predict(model, gen):
    y_true = gen.labels.astype(np.int32)
    y_prob = model.predict(gen, verbose=0).reshape(-1)
    return y_true, y_prob

def _plot_roc(y_true, y_prob, out_path=PLOT_ROC_PATH):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    return roc_auc

def _plot_cm(y_true, y_pred_bin, out_path=PLOT_CM_PATH):
    cm = confusion_matrix(y_true, y_pred_bin)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
    return cm

def evaluate(threshold: float = 0.5, checkpoint_path: str = CHECKPOINT_PATH):
    logger = get_logger()
    ensure_dirs(RESULTS_DIR)

    _, val_gen, test_gen = create_data_generators()

    logger.info(f"Loading model: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                           tf.keras.metrics.AUC(curve="ROC", name="auc"),
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])

    # Validation
    y_true_val, y_prob_val = _predict(model, val_gen)
    roc_auc_val = _plot_roc(y_true_val, y_prob_val)
    y_pred_val = (y_prob_val >= threshold).astype(int)
    cm_val = _plot_cm(y_true_val, y_pred_val)

    # Test
    y_true_test, y_prob_test = _predict(model, test_gen)
    fpr, tpr, _ = roc_curve(y_true_test, y_prob_test)
    roc_auc_test = auc(fpr, tpr)
    y_pred_test = (y_prob_test >= threshold).astype(int)
    cm_test = confusion_matrix(y_true_test, y_pred_test)

    logger.info(f"Validation ROC AUC: {roc_auc_val:.4f}")
    logger.info(f"Test ROC AUC: {roc_auc_test:.4f}")
    logger.info(f"Val CM:\n{cm_val}")
    logger.info(f"Test CM:\n{cm_test}")

    return {
        "val": {"roc_auc": float(roc_auc_val), "cm": cm_val.tolist()},
        "test": {"roc_auc": float(roc_auc_test), "cm": cm_test.tolist()},
    }

if __name__ == "__main__":
    evaluate()
