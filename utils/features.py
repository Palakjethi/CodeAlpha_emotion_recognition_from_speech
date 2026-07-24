"""
Audio feature extraction for Speech Emotion Recognition.

Extracts MFCC + delta + delta-delta coefficients, fixed to a constant
time length via padding/truncation so all samples can be batched.
"""

import librosa
import numpy as np

SR = 22050          # target sample rate
N_MFCC = 40          # number of MFCC coefficients
MAX_LEN = 173        # ~2.5-3s of audio at hop_length=512, sr=22050 -> adjust as needed
HOP_LENGTH = 512


def load_audio(file_path, sr=SR, duration=3.0, offset=0.0):
    """Load and trim/pad raw audio to a fixed duration (seconds)."""
    y, orig_sr = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr


def extract_mfcc(file_path, n_mfcc=N_MFCC, max_len=MAX_LEN, augment=False):
    """
    Returns a (max_len, n_mfcc*3) feature matrix:
    MFCC + delta + delta-delta, time-major (ready for CNN1d/LSTM: (T, C)).
    """
    y, sr = load_audio(file_path)

    if augment:
        y = _augment(y, sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([mfcc, delta, delta2], axis=0)  # (n_mfcc*3, T)

    if features.shape[1] < max_len:
        pad = max_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad)), mode="constant")
    else:
        features = features[:, :max_len]

    return features.T.astype(np.float32)  # (max_len, n_mfcc*3)


def _augment(y, sr):
    """Random augmentation: pitch shift, time stretch, or add noise (pick one)."""
    choice = np.random.choice(["none", "noise", "pitch", "stretch"], p=[0.4, 0.2, 0.2, 0.2])
    if choice == "noise":
        noise_amp = 0.01 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
    elif choice == "pitch":
        steps = np.random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    elif choice == "stretch":
        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)
        # re-fix length after stretch
        target_len = sr * 3
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
    return y


def normalize_features(X, mean=None, std=None):
    """Z-score normalize features. Pass mean/std from train set when applying to val/test."""
    if mean is None:
        mean = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X - mean) / std, mean, std
