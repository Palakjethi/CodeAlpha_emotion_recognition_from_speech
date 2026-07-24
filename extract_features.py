"""
Step 2: Extract MFCC (+delta+delta2) features for every file in labels.csv
and cache them to a single .npz file for fast training.

Usage:
    python extract_features.py
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.features import extract_mfcc
from utils.label_parser import UNIFIED_LABELS

LABELS_CSV = "data/labels.csv"
OUTPUT_NPZ = "data/features.npz"

label2idx = {label: i for i, label in enumerate(UNIFIED_LABELS)}


def main():
    df = pd.read_csv(LABELS_CSV)
    print(f"Extracting features for {len(df)} files...")

    X, y, speakers, datasets = [], [], [], []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            feat = extract_mfcc(row["path"])
            X.append(feat)
            y.append(label2idx[row["emotion"]])
            speakers.append(row["speaker"])
            datasets.append(row["dataset"])
        except Exception as e:
            failed += 1
            continue

    X = np.stack(X)                # (N, T, F)
    y = np.array(y, dtype=np.int64)
    speakers = np.array(speakers)
    datasets = np.array(datasets)

    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, X=X, y=y, speakers=speakers, datasets=datasets)

    print(f"\nDone. Saved features to {OUTPUT_NPZ}")
    print(f"X shape: {X.shape}  (samples, time_steps, features)")
    print(f"Failed to process: {failed} files")
    print(f"Label mapping: {label2idx}")


if __name__ == "__main__":
    main()
