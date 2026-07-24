# Speech Emotion Recognition (SER)

A deep learning system that recognizes human emotions (angry, happy, sad, neutral,
fear, disgust, surprise) from speech audio, using MFCC features and a CNN-BiLSTM
model with attention.

Built as part of the CodeAlpha internship.

---

## Overview

| | |
|---|---|
| **Task** | Classify speech audio into 7 emotions |
| **Datasets used** | RAVDESS, TESS, EMO-DB (Berlin) — combined |
| **Total samples** | 4,775 labeled audio clips across 37 speakers |
| **Features** | MFCC + Delta + Delta-Delta (120-dim, 173 time-steps per clip) |
| **Model** | CNN + BiLSTM + Attention |
| **Final test performance** | 61% accuracy, 0.58 macro-F1 (7-class) |

---

## 1. Datasets

Three public emotional speech datasets were combined to increase data diversity
and reduce overfitting to any single set of speakers:

| Dataset | Files used | Speakers | Language | Notes |
|---|---|---|---|---|
| **RAVDESS** | 1,440 | 24 (12M/12F) | English | Professional actors, North American accent |
| **TESS** | 2,800 | 2 (F) | English | Two actresses (young + older), very clean audio |
| **EMO-DB** (Berlin) | 535 | ~10 | German | Classic SER benchmark, smaller dataset |
| **Total** | **4,775** | **37** | mixed | |

### Unified label set (7 classes)

Since each dataset labels emotions differently, all labels were mapped to a
common set of 7 classes:

```
neutral, happy, sad, angry, fear, disgust, surprise
```

- RAVDESS's `calm` category was merged into `neutral`
- EMO-DB's `boredom` category was also merged into `neutral`
- TESS's `pleasant_surprise` was mapped to `surprise`

### Final class distribution (after merging)

```
neutral     848
angry       719
happy       663
fear        661
sad         654
disgust     638
surprise    592
```

### Filename parsing per dataset

Each dataset encodes the emotion label directly in its filenames using a
different convention:

- **RAVDESS**: `03-01-06-01-02-01-12.wav` → 3rd number = emotion code
- **TESS**: `OAF_back_angry.wav` → emotion is the last word in the filename
- **EMO-DB**: `03a01Fa.wav` → 6th character = emotion code letter

This parsing logic lives in `utils/label_parser.py`.

---

## 2. Feature Extraction: MFCC

Rather than feeding raw audio into the model, each clip was converted into a
numeric representation using **MFCC (Mel-Frequency Cepstral Coefficients)** —
a standard technique in speech processing that approximates how the human ear
perceives sound.

**Pipeline** (`utils/features.py`):
1. Load audio, resample to 22,050 Hz
2. Trim/pad every clip to a fixed 3-second duration
3. Extract 40 MFCC coefficients per frame
4. Extract **Delta** (rate of change) and **Delta-Delta** (acceleration) of the MFCCs
   — these capture how the sound changes over time, not just a static snapshot
5. Concatenate: 40 MFCC + 40 Delta + 40 Delta-Delta = **120 features per time-step**
6. Pad/truncate every clip to **173 time-steps**

**Result:** every audio clip becomes a fixed-size `(173, 120)` matrix.
Across all 4,775 clips, this produced a feature array of shape:

```
X shape: (4775, 173, 120)
```

with **zero failed extractions**. This was cached to `data/features.npz` so
feature extraction only needs to run once.

---

## 3. Model: CNN-BiLSTM with Attention

Defined in `models/cnn_lstm.py`.

```
Input: (batch, 173, 120)  MFCC+delta+delta2 features
   ↓
Conv1D (128 filters) + BatchNorm + ReLU + MaxPool + Dropout
   ↓
Conv1D (256 filters) + BatchNorm + ReLU + MaxPool + Dropout
   ↓
BiLSTM (2 layers, 128 hidden units, bidirectional)
   ↓
Attention layer (learns to weight the most emotionally salient time frames)
   ↓
Dense(128) + ReLU + Dropout
   ↓
Dense(7) → Softmax  (7 emotion classes)
```

- **CNN layers** extract local spectral patterns (e.g. formant shapes) from the MFCCs
- **BiLSTM** captures how those patterns evolve over the course of the utterance,
  reading both forward and backward in time
- **Attention** lets the model focus more on the time-frames that carry the
  strongest emotional signal, rather than weighting the whole clip equally

Total trainable parameters: **~1.07 million**

---

## 4. Training Setup

Defined in `train.py`.

- **Speaker-level train/val/test split** (not random sample split) — this is
  important: it ensures the model is tested on voices it has *never heard*
  during training, giving an honest measure of generalization rather than
  letting it partially "memorize" a speaker's voice.
  - Train: 2,710 clips
  - Validation: 371 clips
  - Test: 1,694 clips
  - 37 unique speakers total, split at the speaker level

- **Class-weighted loss** (Cross-Entropy) to handle mild class imbalance
- **Adam optimizer** with learning-rate scheduling (`ReduceLROnPlateau`)
- **Early stopping** (patience = 10 epochs) based on validation macro-F1
- **Gradient clipping** to stabilize training
- Training stopped automatically at **epoch 29** via early stopping

---

## 5. Results

### Final test set performance (1,694 held-out clips, unseen speakers)

```
Test accuracy: 61%
Test macro-F1: 0.58
```

| Emotion | Precision | Recall | F1-score |
|---|---|---|---|
| fear | 0.66 | 0.87 | **0.75** |
| disgust | 0.88 | 0.65 | **0.74** |
| surprise | 0.55 | 0.90 | 0.68 |
| sad | 0.51 | 0.90 | 0.65 |
| neutral | 0.75 | 0.47 | 0.58 |
| angry | 0.84 | 0.32 | 0.46 |
| happy | 0.26 | 0.16 | **0.20** |

### Key findings

- **Fear and disgust** were recognized most reliably — these emotions have
  fairly distinctive acoustic signatures (tense/breathy for fear, harsh/low
  for disgust).
- **Happy was the weakest class**, frequently confused with **surprise**.
  This is a known, documented difficulty in SER literature: "pleasant
  surprise" and "happy" speech share very similar acoustic properties
  (high pitch, high energy, upbeat tone), especially in acted datasets like TESS.
- **Angry** had high precision (0.84) but low recall (0.32) — when the model
  says "angry" it's usually right, but it misses many true angry clips,
  often confusing them with **fear** (also a high-arousal, tense-sounding emotion).
- Performance on **speakers seen during training** was consistently much
  higher (98-99%+ confidence, correct) than on genuinely unseen speakers —
  demonstrating why the speaker-level split matters for honest evaluation.

---

## 6. Project Structure

```
ser_project/
├── data/
│   ├── raw/                # downloaded datasets go here (not included in repo — see setup)
│   │   ├── RAVDESS/
│   │   ├── TESS/
│   │   └── EMODB/
│   ├── labels.csv           # generated by build_dataset.py (not included — regenerate locally)
│   └── features.npz         # generated by extract_features.py (not included — regenerate locally)
├── models/
│   ├── cnn_lstm.py          # model architecture
│   └── best_model.pt        # trained weights (included in repo)
├── utils/
│   ├── label_parser.py      # unifies emotion labels across datasets
│   └── features.py          # MFCC + delta + delta-delta extraction
├── build_dataset.py         # Step 1: scan raw audio, build labels.csv
├── extract_features.py      # Step 2: extract & cache MFCC features
├── train.py                 # Step 3: train the CNN-BiLSTM model
├── predict.py                # Step 4: run inference on a new .wav file
└── requirements.txt
```

---

## 7. Setup & Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download the datasets (required to retrain — not included in this repo due to size)

| Dataset | Source |
|---|---|
| RAVDESS | https://zenodo.org/record/1188976 (`Audio_Speech_Actors_01-24.zip`) |
| TESS | https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess |
| EMO-DB | http://emodb.bilderbar.info/download/ (use the `wav/` folder only) |

Place them under `data/raw/RAVDESS/`, `data/raw/TESS/`, `data/raw/EMODB/wav/`
respectively, keeping original filenames intact.

### Run the full pipeline

```bash
python build_dataset.py         # -> data/labels.csv
python extract_features.py      # -> data/features.npz
python train.py                 # -> models/best_model.pt
```

### Run inference on your own audio (works immediately, using the included trained model)

```bash
python predict.py path/to/your_audio.wav
```

---

## 8. Possible future improvements

- Fine-tune a pretrained self-supervised speech model (Wav2Vec2 / HuBERT)
  instead of training MFCC-based features from scratch — current
  state-of-the-art SER systems using these reach 0.7-0.8 macro-F1
- Data augmentation (pitch shift, time stretch, noise injection) — implemented
  in `utils/features.py` but not used in the final training run
- Address the happy/surprise confusion specifically, e.g. by collecting
  more spontaneous (non-acted) emotional speech data
- Cross-corpus evaluation (train on some datasets, test purely on a held-out
  dataset) to measure generalization across recording conditions/languages
