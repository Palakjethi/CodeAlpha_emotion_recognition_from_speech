"""
Step 1: Scan raw dataset folders and build a unified labels.csv

Usage:
    python build_dataset.py
"""

import os
import glob
import pandas as pd
from utils.label_parser import parse_label

RAW_DIR = "data/raw"
OUTPUT_CSV = "data/labels.csv"

DATASETS = {
    "RAVDESS": os.path.join(RAW_DIR, "RAVDESS"),
    "TESS": os.path.join(RAW_DIR, "TESS"),
    "EMODB": os.path.join(RAW_DIR, "EMODB"),
}


def scan_dataset(dataset_name, root_dir):
    rows = []
    if not os.path.isdir(root_dir):
        print(f"[WARN] {root_dir} not found, skipping {dataset_name}. "
              f"Download it and place files there (see README.md).")
        return rows

    wav_files = glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
    for f in wav_files:
        emotion, speaker = parse_label(f, dataset_name)
        if emotion is None:
            continue
        rows.append({
            "path": os.path.abspath(f),
            "dataset": dataset_name,
            "emotion": emotion,
            "speaker": speaker,
        })
    print(f"[{dataset_name}] found {len(wav_files)} wav files, "
          f"{len(rows)} labeled successfully.")
    return rows


def main():
    all_rows = []
    for name, path in DATASETS.items():
        all_rows.extend(scan_dataset(name, path))

    if not all_rows:
        print("\nNo data found. Please download the datasets first (see README.md) "
              "and place them under data/raw/<DATASET_NAME>/")
        return

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df)} labeled samples to {OUTPUT_CSV}")
    print("\nClass distribution:")
    print(df["emotion"].value_counts())
    print("\nPer-dataset counts:")
    print(df["dataset"].value_counts())
    print(f"\nUnique speakers: {df['speaker'].nunique()}")


if __name__ == "__main__":
    main()
