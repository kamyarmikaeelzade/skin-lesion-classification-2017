# src/models.py
"""
Model zoo with selectable Keras backbones:
EfficientNetB0/B3, ResNet50, DenseNet121, MobileNetV2, InceptionV3, Xception.

All use a simple GAP -> Dropout -> Dense(1, sigmoid) head for binary classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB3,
    ResNet50, DenseNet121, MobileNetV2,
    InceptionV3, Xception
)

from .config import IMG_SIZE, NUM_CLASSES, DROPOUT, FREEZE_BACKBONE, LEARNING_RATE, MIXED_PRECISION

_BACKBONES = {
    "EfficientNetB0": EfficientNetB0,
    "EfficientNetB3": EfficientNetB3,
    "ResNet50": ResNet50,
    "DenseNet121": DenseNet121,
    "MobileNetV2": MobileNetV2,
    "InceptionV3": InceptionV3,
    "Xception": Xception,
}

def list_backbones():
    return list(_BACKBONES.keys())

def _backbone(name: str, inputs):
    if name not in _BACKBONES:
        raise ValueError(f"Unsupported backbone: {name}. Options: {list_backbones()}")
    return _BACKBONES[name](include_top=False, weights="imagenet", input_tensor=inputs)

def build_model(
    img_size=IMG_SIZE,
    backbone_name: str = "EfficientNetB0",
    num_classes: int = NUM_CLASSES,
    dropout: float = DROPOUT,
    freeze_backbone: bool = FREEZE_BACKBONE,
    learning_rate: float = LEARNING_RATE
):
    if MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    base = _backbone(backbone_name, inputs)

    if freeze_backbone:
        for l in base.layers:
            l.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout)(x)

    # Binary classifier head (keep dtype float32 when mixed precision)
    dtype = "float32" if MIXED_PRECISION else None
    outputs = layers.Dense(1, activation="sigmoid", dtype=dtype)(x)

    model = models.Model(inputs, outputs, name=f"{backbone_name}_melanoma_classifier")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(curve="ROC", name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
