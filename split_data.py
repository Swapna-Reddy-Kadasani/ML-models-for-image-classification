"""Split cats and dogs images into train / validation / test sets."""
import os
import shutil
import random
from config import (
    PROJECT_ROOT, RAW_CATS, RAW_DOGS, DATA_ROOT,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE, CLASS_NAMES,
)

def main():
    random.seed(RANDOM_STATE)
    os.makedirs(DATA_ROOT, exist_ok=True)
    for split in ("train", "val", "test"):
        for name in CLASS_NAMES:
            os.makedirs(os.path.join(DATA_ROOT, split, name), exist_ok=True)

    for class_name, raw_dir in [("cats", RAW_CATS), ("dogs", RAW_DOGS)]:
        if not os.path.isdir(raw_dir):
            print(f"Skip {raw_dir}: not a directory")
            continue
        files = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]
        # Support common image extensions
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val
        splits = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }
        for split, filenames in splits.items():
            dst_dir = os.path.join(DATA_ROOT, split, class_name)
            for f in filenames:
                src = os.path.join(raw_dir, f)
                dst = os.path.join(dst_dir, f)
                if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                    shutil.copy2(src, dst)
        print(f"{class_name}: train={n_train}, val={n_val}, test={n_test}")

    print("Data split done. Structure:", DATA_ROOT)

if __name__ == "__main__":
    main()
