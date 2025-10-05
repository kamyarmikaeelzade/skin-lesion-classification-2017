# src/utils.py
import os
import random
import logging
import numpy as np
import tensorflow as tf

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_logger(name: str = "isic2017"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger
