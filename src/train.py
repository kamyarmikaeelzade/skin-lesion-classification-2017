# src/train.py
"""
Training entrypoint:
- builds chosen backbone
- trains with early stop + best checkpoint
- writes CSV history + TensorBoard logs
- optional fine-tuning (unfreeze)
"""

import argparse
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau, TensorBoard
)

from .config import (
    EPOCHS, CHECKPOINT_PATH, HISTORY_CSV, LOG_DIR, RESULTS_DIR,
    FREEZE_BACKBONE, BACKBONE
)
from .dataset import create_data_generators
from .models import build_model, list_backbones
from .utils import set_global_seed, get_logger, ensure_dirs

def _callbacks():
    return [
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(HISTORY_CSV, append=False),
        TensorBoard(log_dir=LOG_DIR),
    ]

def train(epochs: int = EPOCHS, backbone: str = BACKBONE, unfreeze: bool = not FREEZE_BACKBONE):
    logger = get_logger()
    set_global_seed()
    ensure_dirs(LOG_DIR, RESULTS_DIR)

    logger.info(f"Backbone: {backbone} | Available: {list_backbones()}")

    train_gen, val_gen, _ = create_data_generators()
    model = build_model(backbone_name=backbone)

    # Stage 1: train head (if backbone frozen)
    logger.info("Stage 1: training classification head…")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=_callbacks(),
        verbose=1
    )
    pd.DataFrame(history.history).to_csv(HISTORY_CSV, index=False)

    # Stage 2: optional fine-tuning
    if unfreeze:
        logger.info("Stage 2: unfreezing backbone for fine-tuning…")
        for layer in model.layers:
            layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=model.metrics,
        )
        history_ft = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=max(3, epochs // 3),
            callbacks=_callbacks(),
            verbose=1
        )
        # Append finetune history
        hist_df = pd.read_csv(HISTORY_CSV)
        hist_ft_df = pd.DataFrame(history_ft.history)
        pd.concat([hist_df, hist_ft_df], axis=0, ignore_index=True).to_csv(HISTORY_CSV, index=False)

    logger.info(f"Training complete. Best model at: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--backbone", type=str, default=BACKBONE, help="One of: " + ", ".join(list_backbones()))
    parser.add_argument("--unfreeze", action="store_true", help="Enable fine-tuning by unfreezing the backbone")
    args = parser.parse_args()
    train(epochs=args.epochs, backbone=args.backbone, unfreeze=args.unfreeze)
