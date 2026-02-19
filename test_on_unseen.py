"""
Test all saved models on unseen data only.
Uses the held-out test set (data/test) by default, or a custom folder with cats/ and dogs/ subdirs.
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import MODELS_DIR, RESULTS_DIR, CLASS_NAMES, IMG_SIZE, BATCH_SIZE
from models import get_all_models


def get_unseen_generator(data_path, batch_size=BATCH_SIZE):
    """Build generator for unseen data folder (must contain cats/ and dogs/ subdirs)."""
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data path not found: {data_path}\n"
            "Use a real folder path that contains 'cats/' and 'dogs/' subfolders, "
            "or omit --data to use the default test set (data/test)."
        )
    for name in CLASS_NAMES:
        sub = os.path.join(data_path, name)
        if not os.path.isdir(sub):
            raise FileNotFoundError(f"Expected subfolder {sub} (unseen data must have cats/ and dogs/)")
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        data_path,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=False,
    )


def run_unseen_test(data_path=None):
    """Run all saved models on unseen data and return metrics + confusion matrices."""
    from config import DATA_ROOT
    if data_path is None:
        data_path = os.path.join(DATA_ROOT, "test")
    print("=" * 60)
    print("TESTING ON UNSEEN DATA")
    print("=" * 60)
    print(f"Data path: {os.path.abspath(data_path)}")
    print("(These images were never used for training or validation.)\n")

    gen = get_unseen_generator(data_path)
    n_samples = gen.samples
    print(f"Total samples: {n_samples}\n")

    all_metrics = []
    all_cms = []
    for name in get_all_models().keys():
        path = os.path.join(MODELS_DIR, f"{name}.keras")
        if not os.path.isfile(path):
            print(f"Skip {name}: no saved model at {path}")
            continue
        model = keras.models.load_model(path)
        gen.reset()
        y_true = np.array(gen.classes)
        y_pred_proba = model.predict(gen, verbose=0)
        y_pred = (np.squeeze(y_pred_proba) >= 0.5).astype(int)
        n = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:n], y_pred[:n]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        all_metrics.append({"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
        all_cms.append((name, cm))

        print(f"--- {name} ---")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
        print("Confusion matrix (rows=true, cols=predicted):")
        print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
        print()

    if not all_metrics:
        print("No saved models found. Train first with train.py")
        return

    df = pd.DataFrame(all_metrics).set_index("model")
    print("=" * 60)
    print("UNSEEN DATA — SUMMARY (all models)")
    print("=" * 60)
    print(df.to_string())
    print()

    # Save report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "unseen_data_results.txt")
    with open(report_path, "w") as f:
        f.write("UNSEEN DATA EVALUATION\n")
        f.write("(Images not used in training or validation)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Data path: {os.path.abspath(data_path)}\n")
        f.write(f"Samples: {n_samples}\n\n")
        f.write("METRICS\n")
        f.write(df.to_string() + "\n\n")
        f.write("CONFUSION MATRICES (rows=true, cols=predicted)\n")
        for name, cm in all_cms:
            f.write(f"\n{name}:\n")
            f.write(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string() + "\n")
    print(f"Report saved to {report_path}")
    return df, all_cms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test models on unseen data")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to folder with cats/ and dogs/ subdirs (default: data/test). Example: --data C:\\Users\\You\\my_images",
    )
    args = parser.parse_args()
    if args.data is not None and args.data.strip().lower() in ("path/to/folder", ""):
        args.data = None  # use default
    run_unseen_test(data_path=args.data)
