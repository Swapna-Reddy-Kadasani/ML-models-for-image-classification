"""Train all classification models with validation, fine-tuning, and checkpointing."""
import os
import tensorflow as tf
from tensorflow import keras
from config import (
    MODELS_DIR, EPOCHS, PATIENCE,
    FINE_TUNE_EPOCHS, FINE_TUNE_LR, FINE_TUNE_PATIENCE, FINE_TUNE_UNFREEZE_LAST_LAYERS,
)
from data_pipeline import get_train_val_generators
from models import get_all_models, prepare_for_fine_tuning


def train_one(name, model, train_gen, val_gen, lr=1e-3, epochs=EPOCHS, patience=PATIENCE, save_path=None):
    """Train a single model and save best weights."""
    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"{name}.keras")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    os.makedirs(MODELS_DIR, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# Transfer learning model names (get fine-tuning); small_cnn does not
TRANSFER_NAMES = {"resnet50", "efficientnetb0", "mobilenetv2"}


def main():
    print("Loading data...")
    train_gen, val_gen = get_train_val_generators()
    print(f"Train steps: {train_gen.samples}, Val steps: {val_gen.samples}")

    for name, (build_fn, _) in get_all_models().items():
        print("\n" + "=" * 60)
        print(f"Training: {name}")
        print("=" * 60)
        model = build_fn()
        save_path = os.path.join(MODELS_DIR, f"{name}.keras")

        # Phase 1: train head (all models)
        print("Phase 1: training classifier head (frozen backbone for transfer models)...")
        train_one(name, model, train_gen, val_gen, lr=1e-3, epochs=EPOCHS, patience=PATIENCE, save_path=save_path)

        # Phase 2: fine-tune backbone (transfer learning models only)
        if name in TRANSFER_NAMES:
            print("\nPhase 2: fine-tuning backbone (last {} layers, lr={})...".format(
                FINE_TUNE_UNFREEZE_LAST_LAYERS, FINE_TUNE_LR))
            prepare_for_fine_tuning(model, unfreeze_last_n=FINE_TUNE_UNFREEZE_LAST_LAYERS)
            train_one(
                name, model, train_gen, val_gen,
                lr=FINE_TUNE_LR, epochs=FINE_TUNE_EPOCHS, patience=FINE_TUNE_PATIENCE, save_path=save_path,
            )

        print(f"Saved best model to {save_path}")

    print("\nAll models trained. Run evaluate.py for test-set metrics.")


if __name__ == "__main__":
    main()
