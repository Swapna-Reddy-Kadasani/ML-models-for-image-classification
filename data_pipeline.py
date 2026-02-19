"""Data loading and augmentation for cats vs dogs."""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import DATA_ROOT, IMG_SIZE, BATCH_SIZE, CLASS_NAMES


def get_train_datagen():
    """Training data with stronger augmentation to improve generalization and accuracy."""
    return ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.25,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
    )


def get_val_test_datagen():
    """Validation/test: only rescale."""
    return ImageDataGenerator(rescale=1.0 / 255)


def flow_from_split(split, datagen, shuffle=True, batch_size=BATCH_SIZE):
    """Create generator for train/val/test from data/split/cats|dogs."""
    path = os.path.join(DATA_ROOT, split)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Run split_data.py first. Missing: {path}")
    return datagen.flow_from_directory(
        path,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=shuffle,
    )


def get_train_val_generators():
    """Train and validation generators with correct preprocessing."""
    train_datagen = get_train_datagen()
    val_datagen = get_val_test_datagen()
    train_gen = flow_from_split("train", train_datagen, shuffle=True)
    val_gen = flow_from_split("val", val_datagen, shuffle=False)
    return train_gen, val_gen


def get_test_generator():
    """Test generator (no shuffle for reproducible metrics)."""
    datagen = get_val_test_datagen()
    return flow_from_split("test", datagen, shuffle=False)
