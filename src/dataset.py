# src/dataset.py
"""
Data generators without augmentation, using Keras' ImageDataGenerator.
CSV columns expected: image_id, melanoma (0/1). Image files must exist under *_DIR.
"""

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import (
    TRAIN_CSV, VALID_CSV, TEST_CSV,
    TRAIN_DIR, VALID_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, CLASS_MODE
)

def _normalize_and_fix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["melanoma"] = df["melanoma"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    # Ensure extension
    if not df["image_id"].iloc[0].lower().endswith(".jpg"):
        df["image_id"] = df["image_id"] + ".jpg"
    return df

def create_data_generators(
    train_csv: str = TRAIN_CSV,
    valid_csv: str = VALID_CSV,
    test_csv: str = TEST_CSV,
    train_dir: str = TRAIN_DIR,
    valid_dir: str = VALID_DIR,
    test_dir: str = TEST_DIR,
    batch_size: int = BATCH_SIZE,
    img_size: tuple = IMG_SIZE,
):
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_df = _normalize_and_fix(pd.read_csv(train_csv))
    valid_df = _normalize_and_fix(pd.read_csv(valid_csv))
    test_df  = _normalize_and_fix(pd.read_csv(test_csv))

    print(f"Samples — train: {len(train_df)}, val: {len(valid_df)}, test: {len(test_df)}")

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col="image_id",
        y_col="melanoma",
        target_size=img_size,
        batch_size=batch_size,
        class_mode=CLASS_MODE,
        shuffle=True
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=valid_dir,
        x_col="image_id",
        y_col="melanoma",
        target_size=img_size,
        batch_size=batch_size,
        class_mode=CLASS_MODE,
        shuffle=False
    )

    test_gen = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=test_dir,
        x_col="image_id",
        y_col="melanoma",
        target_size=img_size,
        batch_size=batch_size,
        class_mode=CLASS_MODE,
        shuffle=False
    )

    return train_gen, val_gen, test_gen
