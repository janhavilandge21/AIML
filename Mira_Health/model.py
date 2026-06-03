"""
model.py — Health Prediction Model (Random Forest Classifier)
-------------------------------------------------------------
Exposes a single public function:  predict_health(glucose, haemoglobin, cholesterol)

The model is trained once via train_model.py and saved to disk as
'health_model.pkl'.  If the pickle file is missing when predict_health()
is called, the model is retrained on-the-fly from synthetic data so the
application never crashes.
"""

import os
import pickle
import numpy as np
from typing import Optional

# ─── Paths ───────────────────────────────────────────────────────────────────

MODEL_PATH = "health_model.pkl"


# ─── Clinical reference ranges ───────────────────────────────────────────────
#
#  Glucose      : Normal  70–100 mg/dL  |  Pre-diabetic 100–125  |  Diabetic ≥126
#  Haemoglobin  : Normal  Men ≥13.5 g/dL, Women ≥12.0 g/dL  |  Anaemia <12
#  Cholesterol  : Desirable <200 mg/dL  |  Borderline 200–239   |  High ≥240
#
#  We use gender-neutral conservative thresholds suitable for a demo application.

GLUCOSE_NORMAL_MAX   = 100.0   # mg/dL
GLUCOSE_PREDIAB_MAX  = 125.0   # mg/dL

HAEMO_ANAEMIA        = 12.0    # g/dL  — below this → anaemia risk

CHOL_DESIRABLE_MAX   = 199.9   # mg/dL
CHOL_BORDERLINE_MAX  = 239.9   # mg/dL


# ─── Label encoding ──────────────────────────────────────────────────────────
#
#  The classifier outputs an integer class.  Map it to a human-readable remark.

LABEL_MAP = {
    0: "✅ Healthy — All blood markers within normal range.",
    1: "⚠️ Diabetes Risk — Elevated glucose levels detected. Consult a physician.",
    2: "⚠️ Anaemia Risk — Low haemoglobin levels detected. Dietary review recommended.",
    3: "⚠️ Cardiovascular Risk — High cholesterol detected. Lifestyle changes advised.",
    4: "⚠️ Multiple Risks — Elevated glucose and cholesterol. Medical evaluation needed.",
    5: "⚠️ Diabetes & Anaemia Risk — High glucose and low haemoglobin detected.",
    6: "⚠️ Anaemia & Cardiovascular Risk — Low haemoglobin and high cholesterol.",
    7: "🚨 High Risk — Abnormal glucose, haemoglobin, and cholesterol. Seek medical attention.",
}


# ─── Synthetic dataset generation ────────────────────────────────────────────

def _generate_training_data(n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a labelled synthetic dataset for training.

    Features  : [glucose, haemoglobin, cholesterol]
    Labels    : 0–7  (see LABEL_MAP)

    The label is derived deterministically from clinical thresholds so the
    model learns a meaningful, explainable decision boundary.
    """
    rng = np.random.default_rng(seed=42)

    # ── Healthy samples (class 0) ──────────────────────────────────────────
    n_healthy = n_samples // 4
    glucose_h   = rng.uniform(70,  100, n_healthy)
    haemo_h     = rng.uniform(12.5, 17, n_healthy)
    chol_h      = rng.uniform(130, 199, n_healthy)

    # ── Diabetic risk (class 1) ────────────────────────────────────────────
    n_d = n_samples // 8
    glucose_d   = rng.uniform(126, 250, n_d)
    haemo_d     = rng.uniform(12.5, 17, n_d)
    chol_d      = rng.uniform(130, 199, n_d)

    # ── Anaemia risk (class 2) ────────────────────────────────────────────
    n_a = n_samples // 8
    glucose_a   = rng.uniform(70,  100, n_a)
    haemo_a     = rng.uniform(7.0,  11.9, n_a)
    chol_a      = rng.uniform(130, 199, n_a)

    # ── Cardiovascular risk (class 3) ─────────────────────────────────────
    n_c = n_samples // 8
    glucose_c   = rng.uniform(70,  100, n_c)
    haemo_c     = rng.uniform(12.5, 17, n_c)
    chol_c      = rng.uniform(240, 320, n_c)

    # ── Diabetes + Cardiovascular (class 4) ───────────────────────────────
    n_dc = n_samples // 8
    glucose_dc  = rng.uniform(126, 250, n_dc)
    haemo_dc    = rng.uniform(12.5, 17, n_dc)
    chol_dc     = rng.uniform(240, 320, n_dc)

    # ── Diabetes + Anaemia (class 5) ──────────────────────────────────────
    n_da = n_samples // 8
    glucose_da  = rng.uniform(126, 250, n_da)
    haemo_da    = rng.uniform(7.0,  11.9, n_da)
    chol_da     = rng.uniform(130, 199, n_da)

    # ── Anaemia + Cardiovascular (class 6) ────────────────────────────────
    n_ac = n_samples // 8
    glucose_ac  = rng.uniform(70,  100, n_ac)
    haemo_ac    = rng.uniform(7.0,  11.9, n_ac)
    chol_ac     = rng.uniform(240, 320, n_ac)

    # ── All three (class 7) ───────────────────────────────────────────────
    n_all = n_samples // 8
    glucose_all = rng.uniform(126, 250, n_all)
    haemo_all   = rng.uniform(7.0,  11.9, n_all)
    chol_all    = rng.uniform(240, 320, n_all)

    # ── Concatenate ───────────────────────────────────────────────────────
    X = np.column_stack([
        np.concatenate([glucose_h,  glucose_d,  glucose_a,  glucose_c,
                        glucose_dc, glucose_da, glucose_ac, glucose_all]),
        np.concatenate([haemo_h,    haemo_d,    haemo_a,    haemo_c,
                        haemo_dc,   haemo_da,   haemo_ac,   haemo_all]),
        np.concatenate([chol_h,     chol_d,     chol_a,     chol_c,
                        chol_dc,    chol_da,    chol_ac,    chol_all]),
    ])

    y_parts = [
        np.zeros(n_healthy, dtype=int),
        np.ones(n_d,  dtype=int) * 1,
        np.ones(n_a,  dtype=int) * 2,
        np.ones(n_c,  dtype=int) * 3,
        np.ones(n_dc, dtype=int) * 4,
        np.ones(n_da, dtype=int) * 5,
        np.ones(n_ac, dtype=int) * 6,
        np.ones(n_all,dtype=int) * 7,
    ]
    y = np.concatenate(y_parts)

    # ── Add Gaussian noise to prevent perfect linearity ───────────────────
    X[:, 0] += rng.normal(0, 3,   len(X))   # glucose noise ±3 mg/dL
    X[:, 1] += rng.normal(0, 0.2, len(X))   # haemo  noise ±0.2 g/dL
    X[:, 2] += rng.normal(0, 5,   len(X))   # chol   noise ±5 mg/dL

    return X, y


# ─── Training ────────────────────────────────────────────────────────────────

def train_and_save() -> None:
    """
    Train a Random Forest on synthetic clinical data and pickle the model.

    Why Random Forest?
    ------------------
    • Handles non-linear decision boundaries without feature scaling.
    • Built-in feature importance — interpretable for healthcare stakeholders.
    • Robust against noisy / missing-value patterns in real-world lab results.
    • Low hyper-parameter sensitivity → works well out-of-the-box.
    """
    # Lazy import to keep startup fast when the model already exists
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("🔬 Generating synthetic training data …")
    X, y = _generate_training_data(n_samples=2000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🌳 Training Random Forest Classifier …")
    clf = RandomForestClassifier(
        n_estimators=150,    # number of trees — more = better, slower
        max_depth=10,        # prevent over-fitting on small datasets
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,           # use all CPU cores
    )
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"✅ Model trained — Test accuracy: {acc * 100:.1f}%")

    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(clf, fh)

    print(f"💾 Model saved to '{MODEL_PATH}'")


# ─── Prediction ──────────────────────────────────────────────────────────────

def _load_model():
    """Load the persisted model, training it first if the pickle is absent."""
    if not os.path.exists(MODEL_PATH):
        print("⚙️  Model not found — training now …")
        train_and_save()

    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


def predict_health(glucose: float, haemoglobin: float, cholesterol: float) -> str:
    """
    Predict a health condition from three blood-test values.

    Parameters
    ----------
    glucose      : blood glucose in mg/dL
    haemoglobin  : haemoglobin in g/dL
    cholesterol  : total cholesterol in mg/dL

    Returns
    -------
    Human-readable prediction string from LABEL_MAP.
    """
    try:
        clf = _load_model()
        features = np.array([[glucose, haemoglobin, cholesterol]], dtype=float)
        label_id = int(clf.predict(features)[0])
        return LABEL_MAP.get(label_id, "⚠️ Unknown condition — please consult a doctor.")
    except Exception as exc:
        # Fallback to rule-based logic so the UI never breaks
        return _rule_based_fallback(glucose, haemoglobin, cholesterol)


def _rule_based_fallback(glucose: float, haemoglobin: float, cholesterol: float) -> str:
    """
    Simple threshold-based fallback in case model loading fails.
    Mirrors the training labels so output is consistent.
    """
    high_glucose = glucose >= 126
    low_haemo    = haemoglobin < 12.0
    high_chol    = cholesterol >= 240

    if high_glucose and low_haemo and high_chol:
        return LABEL_MAP[7]
    elif high_glucose and low_haemo:
        return LABEL_MAP[5]
    elif high_glucose and high_chol:
        return LABEL_MAP[4]
    elif low_haemo and high_chol:
        return LABEL_MAP[6]
    elif high_glucose:
        return LABEL_MAP[1]
    elif low_haemo:
        return LABEL_MAP[2]
    elif high_chol:
        return LABEL_MAP[3]
    else:
        return LABEL_MAP[0]
