"""Configuration for cats vs dogs classification."""
import os

# Paths (dataset has cats/ and dogs/ at project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_CATS = os.path.join(PROJECT_ROOT, "cats")
RAW_DOGS = os.path.join(PROJECT_ROOT, "dogs")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")  # train/val/test after split
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Splits (train / val / test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# Image and training
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
PATIENCE = 5  # early stopping

# Fine-tuning (transfer learning models only): phase 2 after training head
FINE_TUNE_EPOCHS = 15
FINE_TUNE_LR = 1e-5
FINE_TUNE_PATIENCE = 4
# Unfreeze last N layers of backbone (None = unfreeze all backbone layers)
FINE_TUNE_UNFREEZE_LAST_LAYERS = 30

# Class names
CLASS_NAMES = ["cats", "dogs"]
