"""Image classification models for cats vs dogs."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from config import IMG_SIZE


def prepare_for_fine_tuning(model, unfreeze_last_n=30):
    """
    Unfreeze the pretrained backbone for fine-tuning.
    Only the last `unfreeze_last_n` layers of the backbone are trainable (rest frozen).
    """
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 1:
            base = layer
            base.trainable = True
            n = len(base.layers)
            for i, l in enumerate(base.layers):
                l.trainable = i >= (n - unfreeze_last_n)
            return
    model.trainable = True


def build_small_cnn(input_shape=(*IMG_SIZE, 3)):
    """Lightweight CNN from scratch (baseline)."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="small_cnn")
    return model


def _add_classifier(base_model, name):
    """Freeze base, add global pool and binary classifier."""
    base_model.trainable = False
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=name)


def build_resnet50():
    """ResNet50 pretrained on ImageNet, top layers for binary classification."""
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    return _add_classifier(base, "resnet50")


def build_efficientnet():
    """EfficientNetB0 pretrained, top for binary classification."""
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    return _add_classifier(base, "efficientnetb0")


def build_mobilenet():
    """MobileNetV2 pretrained, top for binary classification."""
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    return _add_classifier(base, "mobilenetv2")


def get_all_models():
    """Return dict of name -> (build_fn, use_pretrained_preprocess).
    EfficientNet/ResNet/MobileNet expect preprocessing in [0,1] with ImageNet mean/std
    or we use rescale 1/255; for transfer learning rescale is usually enough.
    """
    return {
        "small_cnn": (build_small_cnn, False),
        "resnet50": (build_resnet50, False),
        "efficientnetb0": (build_efficientnet, False),
        "mobilenetv2": (build_mobilenet, False),
    }
