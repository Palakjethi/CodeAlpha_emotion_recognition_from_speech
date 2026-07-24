"""
Step 4: Predict emotion for a single .wav file using the trained model.

Usage:
    python predict.py path/to/audio.wav
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

from models.cnn_lstm import CNN_BiLSTM_SER
from utils.features import extract_mfcc

MODEL_PATH = "models/best_model.pt"
NORM_STATS_PATH = "data/norm_stats.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = CNN_BiLSTM_SER(
        input_dim=checkpoint["input_dim"],
        num_classes=len(checkpoint["label_map"]),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["label_map"]


def predict(file_path):
    model, label_map = load_model()
    norm = np.load(NORM_STATS_PATH)
    mean, std = norm["mean"], norm["std"]

    feat = extract_mfcc(file_path)               # (T, F)
    feat = (feat - mean[0]) / std[0]              # normalize with train stats
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, F)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = probs.argmax()
    print(f"\nFile: {file_path}")
    print(f"Predicted emotion: {label_map[pred_idx]}  (confidence: {probs[pred_idx]:.2%})\n")
    print("All class probabilities:")
    for label, p in sorted(zip(label_map, probs), key=lambda t: -t[1]):
        print(f"  {label:10s}: {p:.2%}")

    return label_map[pred_idx], probs


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/audio.wav")
        sys.exit(1)
    predict(sys.argv[1])
