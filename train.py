"""
Step 3: Train the CNN-BiLSTM Speech Emotion Recognition model.

Uses a SPEAKER-level split (not random sample split) for train/val/test,
so the model is evaluated on voices it has never heard during training.

Usage:
    python train.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from models.cnn_lstm import CNN_BiLSTM_SER
from utils.label_parser import UNIFIED_LABELS

FEATURES_NPZ = "data/features.npz"
MODEL_OUT = "models/best_model.pt"
BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-3
PATIENCE = 10           # early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SERDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def speaker_split(X, y, speakers, test_size=0.15, val_size=0.15, seed=42):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(X, y, groups=speakers))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx, val_idx = next(gss2.split(
        X[trainval_idx], y[trainval_idx], groups=speakers[trainval_idx]
    ))
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    return train_idx, val_idx, test_idx


def normalize(X_train, X_val, X_test):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_labels


def main():
    print(f"Using device: {DEVICE}")
    data = np.load(FEATURES_NPZ, allow_pickle=True)
    X, y, speakers = data["X"], data["y"], data["speakers"]
    print(f"Loaded features: X={X.shape}, y={y.shape}, unique speakers={len(set(speakers))}")

    train_idx, val_idx, test_idx = speaker_split(X, y, speakers)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    X_train, X_val, X_test, mean, std = normalize(X_train, X_val, X_test)
    np.savez("data/norm_stats.npz", mean=mean, std=std)

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    train_loader = DataLoader(SERDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SERDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(SERDataset(X_test, y_test), batch_size=BATCH_SIZE)

    num_classes = len(UNIFIED_LABELS)
    model = CNN_BiLSTM_SER(input_dim=X.shape[2], num_classes=num_classes).to(DEVICE)

    # handle class imbalance
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_val_f1 = 0.0
    epochs_no_improve = 0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "label_map": UNIFIED_LABELS,
                "input_dim": X.shape[2],
            }, MODEL_OUT)
            print(f"  -> saved new best model (val_macro_f1={val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # final test evaluation with best model
    checkpoint = torch.load(MODEL_OUT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_f1, preds, labels = evaluate(model, test_loader, criterion)

    print(f"\n=== TEST RESULTS ===")
    print(f"Test loss: {test_loss:.4f}  Test macro-F1: {test_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=UNIFIED_LABELS, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
