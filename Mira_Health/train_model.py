"""
train_model.py — Standalone model training script
--------------------------------------------------
Run this once before launching the app to pre-train and persist the
Random Forest model to disk.

Usage
-----
    python train_model.py

The script will:
  1. Generate 2 000 synthetic patient records using clinical thresholds.
  2. Split them 80 / 20 into training and test sets.
  3. Fit a Random Forest Classifier.
  4. Print accuracy metrics and a feature-importance table.
  5. Save the trained model to 'health_model.pkl'.

If you skip this step, the Streamlit app will train the model automatically
on first use — this script just makes startup faster.
"""

import sys
import numpy as np


def main() -> None:
    # ── Dependency check ─────────────────────────────────────────────────
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
    except ImportError:
        print("❌  scikit-learn is not installed.")
        print("    Run:  pip install scikit-learn")
        sys.exit(1)

    # ── Import project module ─────────────────────────────────────────────
    from model import _generate_training_data, train_and_save, LABEL_MAP, MODEL_PATH

    print("=" * 60)
    print("  MIRA — Health Prediction Model Training")
    print("=" * 60)

    # ── Generate data ─────────────────────────────────────────────────────
    print("\n[1/4] Generating synthetic dataset …")
    X, y = _generate_training_data(n_samples=2000)
    print(f"      Samples: {len(X)}  |  Features: {X.shape[1]}  |  Classes: {len(np.unique(y))}")

    # ── Split ─────────────────────────────────────────────────────────────
    print("\n[2/4] Splitting 80 / 20 train-test …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n[3/4] Training Random Forest Classifier …")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("      Training complete.")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating …")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Overall Accuracy : {acc * 100:.1f}%")

    label_names = [LABEL_MAP[i].split("—")[0].strip() for i in range(8)]
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Feature importance
    importances = clf.feature_importances_
    feature_names = ["Glucose", "Haemoglobin", "Cholesterol"]
    print("  Feature Importance:")
    for name, imp in zip(feature_names, importances):
        bar = "█" * int(imp * 40)
        print(f"    {name:<13} {imp:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────
    import pickle
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(clf, fh)

    print(f"\n✅  Model saved → '{MODEL_PATH}'")
    print("=" * 60)
    print("  You can now run:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
