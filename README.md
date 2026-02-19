# Cats vs Dogs Image Classification

Binary image classification (cats vs dogs) using multiple deep learning models, with data augmentation and fine-tuning. Built with TensorFlow/Keras.

## Features

- **Models:** Small CNN (from scratch), ResNet50, EfficientNetB0, MobileNetV2 (transfer learning)
- **Pipeline:** Train/validation/test split, augmentation, two-phase training (head + fine-tuning)
- **Evaluation:** Metrics, confusion matrices, and plots on unseen data
- **Testing:** Script to run predictions on a custom folder of images

## Dataset

Place images in two folders at the project root:

- `cats/` – cat images (e.g. .jpg)
- `dogs/` – dog images

The script splits them into train (70%), validation (15%), and test (15%).

## Setup

```bash
cd /path/to/catsdogs
python -m venv myenv
# Windows (PowerShell):
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start

**1. Split data**

```bash
python split_data.py
```

**2. Train all models**

```bash
python train.py
```

**3. Evaluate on test set**

```bash
python evaluate.py
```

**Or run everything:**

```bash
python run_all.py
```

## Test on Unseen Data

To run trained models on your own images (e.g. a folder not used in training):

```bash
python test_on_unseen.py --data "path/to/folder"
```

The folder must contain `cats/` and `dogs/` subfolders. Omit `--data` to use the default test set.

## Project Structure

| File / Folder      | Description                            |
|--------------------|----------------------------------------|
| `config.py`        | Paths, hyperparameters, class names    |
| `split_data.py`    | Creates train/val/test from cats/dogs  |
| `data_pipeline.py` | Data loaders and augmentation         |
| `models.py`        | Model definitions (CNN, ResNet, etc.)  |
| `train.py`         | Training and fine-tuning               |
| `evaluate.py`      | Metrics and confusion matrix plots     |
| `test_on_unseen.py`| Evaluate on a custom image folder      |
| `saved_models/`    | Best model weights (after training)    |
| `results/`         | Metrics, reports, and plots            |

## Results

After training and evaluation, check:

- `results/test_metrics.csv` – accuracy, precision, recall, F1
- `results/all_models_summary.txt` – metrics and confusion matrices
- `results/confusion_matrices_all.png` – confusion matrix plots

MobileNetV2 typically performs best on this setup (~95% test accuracy with augmentation and fine-tuning).

## Configuration

Edit `config.py` to change train/val/test ratios, image size, batch size, epochs, and fine-tuning settings.

## License

Use as you like. Add your own license file if needed.
