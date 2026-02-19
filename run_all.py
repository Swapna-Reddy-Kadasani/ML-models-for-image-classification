"""Run full pipeline: split data -> train all models -> evaluate on test set."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def run(cmd, desc):
    print("\n" + "=" * 60)
    print(desc)
    print("=" * 60)
    r = subprocess.run([sys.executable, cmd], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)

if __name__ == "__main__":
    run("split_data.py", "Step 1: Splitting data into train/val/test")
    run("train.py", "Step 2: Training all models")
    run("evaluate.py", "Step 3: Evaluating on unseen test data")
    print("\nDone. Check the 'results' folder for test_metrics.csv and per-model reports.")
