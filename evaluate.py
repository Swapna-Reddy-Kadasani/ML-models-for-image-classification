"""Evaluate all saved models on the held-out test set and report metrics."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tensorflow import keras
from config import MODELS_DIR, RESULTS_DIR, CLASS_NAMES
from data_pipeline import get_test_generator
from models import get_all_models


def _write_all_models_summary(all_metrics, all_confusion_matrices, results_dir):
    """Write a single file with metrics and confusion matrices for all models."""
    path = os.path.join(results_dir, "all_models_summary.txt")
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ALL MODELS — METRICS & CONFUSION MATRICES (test set, unseen data)\n")
        f.write("=" * 70 + "\n\n")

        # Metrics table
        df = pd.DataFrame(all_metrics).set_index("model")
        df = df[["accuracy", "precision", "recall", "f1"]]
        f.write("METRICS (all models)\n")
        f.write("-" * 50 + "\n")
        f.write(df.to_string() + "\n\n")

        # Confusion matrices (rows = true class, columns = predicted class)
        f.write("CONFUSION MATRICES\n")
        f.write("(rows = true label, columns = predicted label)\n")
        f.write("-" * 50 + "\n")
        for name, cm in all_confusion_matrices:
            cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
            f.write(f"\n{name}:\n")
            f.write(cm_df.to_string() + "\n")
    print(f"Combined summary written to {path}")


def _plot_confusion_matrix_heatmap(cm, class_names, ax, title, cmap="Blues"):
    """Draw a single confusion matrix heatmap on ax."""
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color, fontsize=14)


def _plot_all_confusion_matrices(all_confusion_matrices, results_dir):
    """Save graphical confusion matrices: one combined figure and one per model."""
    n = len(all_confusion_matrices)
    if n == 0:
        return
    # Combined figure (2x2 grid when 4 models)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for idx, (name, cm) in enumerate(all_confusion_matrices):
        ax = axes[idx]
        _plot_confusion_matrix_heatmap(cm, CLASS_NAMES, ax, name, cmap="Blues")
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Confusion matrices (rows = true, cols = predicted)", fontsize=12, y=1.02)
    fig.tight_layout()
    combined_path = os.path.join(results_dir, "confusion_matrices_all.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix plot (all models): {combined_path}")

    # One image per model
    for name, cm in all_confusion_matrices:
        fig, ax = plt.subplots(figsize=(5, 4))
        _plot_confusion_matrix_heatmap(cm, CLASS_NAMES, ax, name, cmap="Blues")
        fig.tight_layout()
        single_path = os.path.join(results_dir, f"confusion_matrix_{name}.png")
        fig.savefig(single_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Per-model confusion matrix plots saved in {results_dir}")


def evaluate_model(name, model, test_gen):
    """Run model on test set and return y_true, y_pred, and metrics dict."""
    test_gen.reset()
    y_true = np.array(test_gen.classes)
    y_pred_proba = model.predict(test_gen, verbose=0)
    y_pred = (np.squeeze(y_pred_proba) >= 0.5).astype(int)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred)
    return y_true, y_pred, metrics, cm


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_gen = get_test_generator()
    print(f"Test samples: {test_gen.samples}")

    all_metrics = []
    all_confusion_matrices = []  # list of (model_name, cm)
    for name in get_all_models().keys():
        path = os.path.join(MODELS_DIR, f"{name}.keras")
        if not os.path.isfile(path):
            print(f"Skip {name}: no saved model at {path}")
            continue
        print(f"\nEvaluating {name}...")
        model = keras.models.load_model(path)
        y_true, y_pred, metrics, cm = evaluate_model(name, model, test_gen)
        metrics["model"] = name
        all_metrics.append(metrics)
        all_confusion_matrices.append((name, cm))
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
        print("Confusion matrix:\n", cm)

        # Save per-model report
        with open(os.path.join(RESULTS_DIR, f"{name}_report.txt"), "w") as f:
            f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
            f.write("\nConfusion matrix:\n")
            f.write(str(cm))

        # Save confusion matrix as CSV (rows=true, cols=predicted)
        cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
        cm_df.to_csv(os.path.join(RESULTS_DIR, f"confusion_matrix_{name}.csv"))

    if not all_metrics:
        print("No models found. Train first with train.py")
        return

    # Summary table (metrics)
    df = pd.DataFrame(all_metrics).set_index("model")
    df = df[["accuracy", "precision", "recall", "f1"]]
    df.to_csv(os.path.join(RESULTS_DIR, "test_metrics.csv"))
    print("\n" + "=" * 60)
    print("Test set performance summary (unseen data)")
    print("=" * 60)
    print(df.to_string())
    print(f"\nResults saved to {RESULTS_DIR}")

    # Write combined summary: all metrics + all confusion matrices
    _write_all_models_summary(all_metrics, all_confusion_matrices, RESULTS_DIR)
    # Graphical confusion matrices
    _plot_all_confusion_matrices(all_confusion_matrices, RESULTS_DIR)


if __name__ == "__main__":
    main()
