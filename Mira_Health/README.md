# 🏥 MIRA — Medical Intelligence Robotic Automation

> A Health Prediction Application built with **Python · Streamlit · SQLite · Scikit-Learn**  

---

## 📌 Project Overview

MIRA is a full-stack health prediction application that allows healthcare staff to:

- Register patients and record their blood-test results (Glucose, Haemoglobin, Cholesterol).
- Automatically generate an **AI-powered health prediction** using a trained Random Forest Classifier.
- Perform complete **CRUD operations** (Create, Read, Update, Delete) on patient records.
- **Search** records by patient name or email.
- View a live **dashboard** with summary statistics.

---

## ✨ Features

| Feature | Details |
|---|---|
| Patient Registration | Full Name, DOB, Email, Glucose, Haemoglobin, Cholesterol |
| AI Health Prediction | Random Forest model trained on synthetic clinical data |
| CRUD Operations | Create · Read · Update (with re-prediction) · Delete |
| Data Validation | Email format · DOB not in future · Numeric blood values · Required fields |
| Persistent Storage | SQLite database (`mira_health.db`) |
| Search | Real-time search by name or email |
| Dashboard | Total patients · Healthy count · At-risk count |
| Clean UI | Streamlit with custom CSS; sidebar navigation |

### Prediction Classes

| Class | Condition |
|---|---|
| 0 | ✅ Healthy |
| 1 | ⚠️ Diabetes Risk (high glucose) |
| 2 | ⚠️ Anaemia Risk (low haemoglobin) |
| 3 | ⚠️ Cardiovascular Risk (high cholesterol) |
| 4 | ⚠️ Multiple Risks — Diabetes + Cardiovascular |
| 5 | ⚠️ Multiple Risks — Diabetes + Anaemia |
| 6 | ⚠️ Multiple Risks — Anaemia + Cardiovascular |
| 7 | 🚨 High Risk — All three markers abnormal |

---

## 🗂️ Project Structure

```
mira_health/
├── app.py             ← Streamlit UI (all pages + navigation)
├── database.py        ← SQLite CRUD layer (init, create, read, update, delete, search)
├── model.py           ← Random Forest predictor (training + inference)
├── train_model.py     ← Standalone training script with evaluation metrics
├── requirements.txt   ← Python package dependencies
├── README.md          ← This file
├── mira_health.db     ← Auto-created SQLite database (git-ignored)
└── health_model.pkl   ← Trained model pickle (auto-created; git-ignored)
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.11** or higher
- `pip` package manager
- A terminal / command prompt

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mira-health-prediction.git
cd mira-health-prediction
```

### Step 2 — (Optional) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — (Optional) Pre-train the model

```bash
python train_model.py
```

> If you skip this step the model is trained automatically when the app first makes a prediction.  
> Pre-training takes ~5 seconds and makes the first prediction instant.

---

## 🚀 Running the Application

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your default browser.

---

## 🖥️ Application Pages

### 🏠 Dashboard
Shows total patient count, healthy count, and at-risk count.  
Displays the 10 most recently registered patients.

### ➕ Add Patient
Registration form with real-time validation.  
On successful submission the AI model runs and the prediction is displayed immediately.

### 📋 Patient Records
Full searchable table of all patients.  
Inline **Edit** expander — update any field and trigger a fresh AI prediction.  
Inline **Delete** expander — confirmation step before permanent removal.

### ℹ️ About
Technology rationale, architecture overview, challenges encountered, and future improvements.

---

## 🤖 AI/ML Details

### Model: Random Forest Classifier
- **Library:** scikit-learn `RandomForestClassifier`
- **Training data:** 2 000 synthetic patient records generated from published clinical thresholds  
  (ADA glucose criteria, WHO haemoglobin cut-offs, ACC/AHA cholesterol guidelines)
- **Features (3):** Glucose · Haemoglobin · Cholesterol
- **Classes (8):** See table above
- **Test accuracy:** ~99 % (synthetic data; clearly separated clinical bands)
- **Serialisation:** `pickle` → `health_model.pkl`

### Why Random Forest?
1. No feature scaling required.
2. Handles non-linear decision boundaries naturally.
3. `feature_importances_` provides interpretable outputs for clinicians.
4. Robust to noisy lab measurements.
5. Sub-millisecond inference — ideal for real-time Streamlit updates.

---

## 🛠️ Technology Stack

| Layer | Technology | Reason |
|---|---|---|
| Frontend / UI | Streamlit | Python-native rapid UI development |
| Backend logic | Python 3.11 | Clean, typed, readable |
| Database | SQLite 3 | Zero-config, file-based, ACID |
| ML model | scikit-learn | Production-grade, interpretable |
| Data wrangling | pandas + numpy | Standard data-science stack |

---

## 📸 Screenshots

_Add screenshots here after recording your demo video._

```
screenshots/
├── dashboard.png
├── add_patient.png
├── patient_records.png
└── about.png
```

---

## 🔮 Future Improvements

1. **Authentication** — JWT / session-based login with doctor vs admin roles.
2. **Real clinical datasets** — retrain on PIMA Indians Diabetes or UCI Heart Disease.
3. **SHAP explainability** — show which feature drove each prediction.
4. **Time-series charts** — per-patient trend view for glucose and cholesterol over time.
5. **REST API** — FastAPI backend to decouple UI from business logic.
6. **Dockerisation** — `Dockerfile` + `docker-compose.yml` for one-command deployment.
7. **PDF reports** — downloadable patient health summary.
8. **Email alerts** — notify the assigned doctor when a patient enters a high-risk category.
9. **PostgreSQL migration** — replace SQLite for multi-user concurrent writes.
10. **Unit tests** — pytest suite covering validation, DB operations, and model inference.

---

## 📧 Submission

- **GitHub Repo:** `https://github.com/YOUR_USERNAME/mira-health-prediction`
- **Demo Video:** Screen recording demonstrating all CRUD operations + AI prediction
- **Email:** jobs@gokulinfocare.com

---

## 📄 Licence

This project was created as a technical assessment submission.  
© 2025 — All rights reserved.

