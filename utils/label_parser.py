"""
Parses raw dataset filenames and returns a unified emotion label.

Unified label set (7 classes):
    neutral, happy, sad, angry, fear, disgust, surprise

RAVDESS 'calm' -> merged into 'neutral'
EMO-DB 'boredom' -> mapped to 'neutral' by default (toggle EMODB_BOREDOM_AS_NEUTRAL)
"""

import os

EMODB_BOREDOM_AS_NEUTRAL = True

UNIFIED_LABELS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

# ---------------------------------------------------------------------------
# RAVDESS
# Filename format: 03-01-06-01-02-01-12.wav
# Position 3 (index 2) = emotion code, position 7 (index 6) = actor id
# ---------------------------------------------------------------------------
_RAVDESS_MAP = {
    "01": "neutral",
    "02": "neutral",   # calm -> neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


def parse_ravdess(filepath):
    fname = os.path.basename(filepath)
    parts = fname.replace(".wav", "").split("-")
    if len(parts) != 7:
        return None, None
    emotion_code = parts[2]
    actor_id = parts[6]
    emotion = _RAVDESS_MAP.get(emotion_code)
    speaker = f"ravdess_actor_{actor_id}"
    return emotion, speaker


# ---------------------------------------------------------------------------
# TESS
# Filename format: OAF_back_angry.wav / YAF_youth_happy.wav
# Emotion is the last "_"-separated token before .wav
# TESS uses 'ps' for pleasant surprise sometimes -> map to 'surprise'
# ---------------------------------------------------------------------------
_TESS_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "ps": "surprise",
    "surprise": "surprise",
    "pleasant_surprise": "surprise",
}


def parse_tess(filepath):
    fname = os.path.basename(filepath).lower().replace(".wav", "")
    tokens = fname.split("_")
    emotion_token = tokens[-1]
    emotion = _TESS_MAP.get(emotion_token)
    # speaker = which actress: folder name usually starts with OAF (older) or YAF (young)
    speaker_prefix = tokens[0] if tokens else "unknown"
    speaker = f"tess_{speaker_prefix}"
    return emotion, speaker


# ---------------------------------------------------------------------------
# EMO-DB (Berlin)
# Filename format: 03a01Fa.wav
#   chars 0-1  -> speaker id (e.g. "03")
#   chars 2-4  -> text code (ignored)
#   char 5     -> emotion code
# ---------------------------------------------------------------------------
_EMODB_MAP = {
    "W": "angry",
    "L": "neutral" if EMODB_BOREDOM_AS_NEUTRAL else "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}


def parse_emodb(filepath):
    fname = os.path.basename(filepath).replace(".wav", "")
    if len(fname) < 6:
        return None, None
    speaker_id = fname[0:2]
    emotion_code = fname[5]
    emotion = _EMODB_MAP.get(emotion_code)
    speaker = f"emodb_{speaker_id}"
    return emotion, speaker


PARSERS = {
    "RAVDESS": parse_ravdess,
    "TESS": parse_tess,
    "EMODB": parse_emodb,
}


def parse_label(filepath, dataset_name):
    """Returns (emotion, speaker_id) or (None, None) if unparseable/unmapped."""
    parser = PARSERS.get(dataset_name)
    if parser is None:
        return None, None
    emotion, speaker = parser(filepath)
    if emotion not in UNIFIED_LABELS:
        return None, None
    return emotion, speaker
