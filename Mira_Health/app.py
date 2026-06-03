"""
app.py — MIRA Health Prediction Application (Streamlit Frontend)
----------------------------------------------------------------
Entry point.  Run with:   streamlit run app.py

Navigation (sidebar):
  🏠 Dashboard      — key stats at a glance
  ➕ Add Patient     — registration form with AI prediction
  📋 Patient Records — searchable data table with inline edit / delete
  ℹ️  About          — project info, tech choices, interview notes
"""

import re
from datetime import date, datetime

import pandas as pd
import streamlit as st

import database as db
from model import predict_health

# ─── Page config (must be first Streamlit call) ──────────────────────────────

st.set_page_config(
    page_title="MIRA — Health Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global typography ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* ── Sidebar ──────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-size: 0.95rem;
    }

    /* ── Metric cards ─────────────────────────────────── */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* ── Buttons ──────────────────────────────────────── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* ── Page header strip ────────────────────────────── */
    .page-header {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: white;
        padding: 20px 28px;
        border-radius: 14px;
        margin-bottom: 24px;
    }
    .page-header h1 { margin: 0; font-size: 1.6rem; }
    .page-header p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.9rem; }

    /* ── Remark badge colours ─────────────────────────── */
    .badge-healthy  { color: #16a34a; font-weight: 600; }
    .badge-warning  { color: #d97706; font-weight: 600; }
    .badge-danger   { color: #dc2626; font-weight: 600; }

    /* ── Data table ───────────────────────────────────── */
    .dataframe { font-size: 0.85rem !important; }

    /* ── Divider ──────────────────────────────────────── */
    hr { border-color: #e2e8f0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Database boot ───────────────────────────────────────────────────────────

db.init_db()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def is_valid_email(email: str) -> bool:
    """Return True if the string matches a basic e-mail pattern."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email.strip()) is not None


def validate_patient_form(
    name: str,
    dob_val,
    email: str,
    glucose: str,
    haemoglobin: str,
    cholesterol: str,
) -> list[str]:
    """
    Validate all form fields.

    Returns a list of human-readable error strings.
    An empty list means validation passed.
    """
    errors: list[str] = []

    # Required fields
    if not name.strip():
        errors.append("Full Name is required.")

    # Email
    if not email.strip():
        errors.append("Email Address is required.")
    elif not is_valid_email(email):
        errors.append("Please enter a valid email address (e.g. user@example.com).")

    # Date of birth — cannot be in the future
    if dob_val is None:
        errors.append("Date of Birth is required.")
    elif dob_val > date.today():
        errors.append("Date of Birth cannot be a future date.")

    # Blood-test values — must be numeric and positive
    for label, raw in [
        ("Glucose",      glucose),
        ("Haemoglobin",  haemoglobin),
        ("Cholesterol",  cholesterol),
    ]:
        raw = str(raw).strip()
        if not raw:
            errors.append(f"{label} is required.")
        else:
            try:
                val = float(raw)
                if val <= 0:
                    errors.append(f"{label} must be a positive number.")
            except ValueError:
                errors.append(f"{label} must be a numeric value.")

    return errors


# ─── Pages ───────────────────────────────────────────────────────────────────

def page_dashboard() -> None:
    """📊 Dashboard page — stats and recent records."""
    st.markdown(
        """
        <div class="page-header">
            <h1>🏥 MIRA Health Prediction Dashboard</h1>
            <p>Medical Intelligence Robotic Automation — Patient Overview</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = db.get_stats()

    # ── Metric cards ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("👥 Total Patients",  stats["total_patients"])
    c2.metric("✅ Healthy",          stats["healthy"])
    c3.metric("⚠️ At Risk",          stats["at_risk"])

    st.markdown("---")

    # ── Recent records ────────────────────────────────────────────────────
    st.subheader("📋 Recent Patient Records")
    patients = db.get_all_patients()

    if not patients:
        st.info("No patients recorded yet.  Use **➕ Add Patient** to register the first record.")
        return

    # Show only the most recent 10 on the dashboard
    df = pd.DataFrame(patients[:10])
    display_cols = ["id", "full_name", "dob", "email", "glucose", "haemoglobin", "cholesterol", "remarks"]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────

def page_add_patient() -> None:
    """➕ Patient registration form."""
    st.markdown(
        """
        <div class="page-header">
            <h1>➕ Register New Patient</h1>
            <p>Enter the patient's details — the AI model will generate a health prediction automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("add_patient_form", clear_on_submit=True):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            name  = st.text_input("Full Name *", placeholder="e.g. Priya Sharma")
            email = st.text_input("Email Address *", placeholder="e.g. priya@example.com")
        with col2:
            dob   = st.date_input(
                "Date of Birth *",
                value=date(1990, 1, 1),
                min_value=date(1900, 1, 1),
                max_value=date.today(),
            )

        st.markdown("---")
        st.subheader("Blood Test Results")

        col3, col4, col5 = st.columns(3)
        with col3:
            glucose     = st.text_input("Glucose (mg/dL) *", placeholder="e.g. 95")
        with col4:
            haemoglobin = st.text_input("Haemoglobin (g/dL) *", placeholder="e.g. 14.5")
        with col5:
            cholesterol = st.text_input("Cholesterol (mg/dL) *", placeholder="e.g. 185")

        st.markdown("---")
        submit = st.form_submit_button("🔬 Analyse & Save Patient", use_container_width=True)

    if submit:
        # ── Validation ───────────────────────────────────────────────────
        errors = validate_patient_form(name, dob, email, glucose, haemoglobin, cholesterol)
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            return

        # ── AI Prediction ─────────────────────────────────────────────────
        with st.spinner("🤖 Running AI health prediction …"):
            remarks = predict_health(float(glucose), float(haemoglobin), float(cholesterol))

        # ── Persist ───────────────────────────────────────────────────────
        ok, msg = db.create_patient(
            full_name=name.strip(),
            dob=str(dob),
            email=email.strip().lower(),
            glucose=float(glucose),
            haemoglobin=float(haemoglobin),
            cholesterol=float(cholesterol),
            remarks=remarks,
        )

        if ok:
            st.success(f"✅ Patient registered successfully!")
            st.info(f"🩺 **AI Prediction:** {remarks}")
        else:
            st.error(f"❌ {msg}")


# ─────────────────────────────────────────────────────────────────────────────

def page_patient_records() -> None:
    """📋 Full patient records with search, edit, and delete."""
    st.markdown(
        """
        <div class="page-header">
            <h1>📋 Patient Records</h1>
            <p>Search, view, update, or delete patient records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Search bar ───────────────────────────────────────────────────────
    search_query = st.text_input(
        "🔍 Search by name or email",
        placeholder="Type a name or email …",
    )

    patients = (
        db.search_patients(search_query)
        if search_query.strip()
        else db.get_all_patients()
    )

    if not patients:
        st.warning("No records found." if search_query else "No patients registered yet.")
        return

    st.caption(f"Showing **{len(patients)}** record(s)")

    # ── Build display DataFrame ───────────────────────────────────────────
    df = pd.DataFrame(patients)
    display_cols = ["id", "full_name", "email", "dob", "glucose", "haemoglobin", "cholesterol", "remarks"]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Edit / Delete expanders ───────────────────────────────────────────
    col_edit, col_del = st.columns(2)

    # ── EDIT ─────────────────────────────────────────────────────────────
    with col_edit:
        with st.expander("✏️  Edit a Patient Record"):
            ids = [p["id"] for p in patients]
            selected_id = st.selectbox(
                "Select Patient ID to edit",
                options=ids,
                key="edit_select",
            )
            patient = db.get_patient_by_id(selected_id)

            if patient:
                with st.form("edit_form"):
                    e_name  = st.text_input("Full Name",       value=patient["full_name"])
                    e_email = st.text_input("Email",           value=patient["email"])
                    e_dob   = st.date_input(
                        "Date of Birth",
                        value=datetime.strptime(patient["dob"], "%Y-%m-%d").date(),
                    )
                    e_glu   = st.text_input("Glucose (mg/dL)", value=str(patient["glucose"]))
                    e_hgb   = st.text_input("Haemoglobin (g/dL)", value=str(patient["haemoglobin"]))
                    e_cho   = st.text_input("Cholesterol (mg/dL)", value=str(patient["cholesterol"]))
                    update_btn = st.form_submit_button("💾 Update & Re-analyse", use_container_width=True)

                if update_btn:
                    errors = validate_patient_form(e_name, e_dob, e_email, e_glu, e_hgb, e_cho)
                    if errors:
                        for err in errors:
                            st.error(f"❌ {err}")
                    else:
                        new_remarks = predict_health(float(e_glu), float(e_hgb), float(e_cho))
                        ok, msg = db.update_patient(
                            patient_id=selected_id,
                            full_name=e_name.strip(),
                            dob=str(e_dob),
                            email=e_email.strip().lower(),
                            glucose=float(e_glu),
                            haemoglobin=float(e_hgb),
                            cholesterol=float(e_cho),
                            remarks=new_remarks,
                        )
                        if ok:
                            st.success(f"✅ {msg}")
                            st.info(f"🩺 **New Prediction:** {new_remarks}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

    # ── DELETE ───────────────────────────────────────────────────────────
    with col_del:
        with st.expander("🗑️  Delete a Patient Record"):
            del_id = st.selectbox(
                "Select Patient ID to delete",
                options=ids,
                key="del_select",
            )
            del_patient = db.get_patient_by_id(del_id)
            if del_patient:
                st.warning(
                    f"You are about to delete **{del_patient['full_name']}** "
                    f"(ID: {del_id}).  This action is irreversible."
                )
            confirm_del = st.button("🗑️ Confirm Delete", type="primary", key="confirm_del")
            if confirm_del:
                ok, msg = db.delete_patient(del_id)
                if ok:
                    st.success(f"✅ {msg}")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")


# ─────────────────────────────────────────────────────────────────────────────

def page_about() -> None:
    """ℹ️ About page — technology choices and interview notes."""
    st.markdown(
        """
        <div class="page-header">
            <h1>ℹ️  About MIRA</h1>
            <p>Technology choices, architecture decisions, and interview notes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🏗️  Project Architecture")
    st.markdown("""
    ```
    mira_health/
    ├── app.py            ← Streamlit UI + page routing
    ├── database.py       ← SQLite CRUD layer
    ├── model.py          ← Random Forest predictor
    ├── train_model.py    ← Standalone training script
    ├── requirements.txt  ← Python dependencies
    └── README.md         ← Full documentation
    ```
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🐍 Why Streamlit?")
        st.markdown("""
        - **Rapid prototyping** — a full CRUD UI requires minimal boilerplate.
        - **Python-native** — no JavaScript context-switching; data scientists stay productive.
        - **Built-in widgets** — forms, tables, charts are first-class citizens.
        - **Deployment simplicity** — `streamlit run app.py` is the only command needed.

        *Trade-off:* Streamlit is not ideal for highly customised UX or high-concurrency production
        apps.  For those, FastAPI + React would be the better choice.
        """)

        st.subheader("🗄️ Why SQLite?")
        st.markdown("""
        - **Zero-config** — no server process, no credentials, no network.
        - **Perfect for demos** — a single `.db` file travels with the project.
        - **Full SQL support** — complex queries, transactions, and ACID guarantees.
        - **Standard library** — no extra pip install; `sqlite3` ships with Python.

        *Trade-off:* SQLite does not scale to high write concurrency.  PostgreSQL or MySQL
        would replace it in a multi-user production deployment.
        """)

    with col2:
        st.subheader("🌳 Why Random Forest?")
        st.markdown("""
        - **No feature scaling** — works with raw mg/dL values.
        - **Handles non-linearity** — real clinical thresholds are rarely a straight line.
        - **Interpretable** — `feature_importances_` gives stakeholders clear explanations.
        - **Robust** — noisy lab results don't break ensemble models.
        - **Fast inference** — prediction in microseconds; ideal for a real-time UI.

        *Trade-off:* For large, high-dimensional EHR datasets, gradient boosting (XGBoost, LightGBM)
        or a neural network would be investigated.
        """)

        st.subheader("🚀 Future Improvements")
        st.markdown("""
        1. **Authentication** — JWT-based login; role-based access (doctor vs admin).
        2. **Real datasets** — retrain on PIMA Indians Diabetes / UCI Heart Disease datasets.
        3. **Explainability** — SHAP values alongside each prediction.
        4. **Charts & trends** — per-patient time-series view for glucose / cholesterol.
        5. **REST API backend** — FastAPI layer; decouple UI from business logic.
        6. **Cloud deployment** — Docker + Streamlit Community Cloud / AWS ECS.
        7. **PDF reports** — one-click patient health summary download.
        8. **Notifications** — email alerts when a patient enters a high-risk category.
        """)

    st.markdown("---")
    st.subheader("⚠️ Challenges Encountered")
    st.markdown("""
    | Challenge | Resolution |
    |-----------|------------|
    | Generating realistic synthetic data without ground-truth labels | Derived labels from published clinical thresholds (ADA glucose criteria, WHO haemoglobin cut-offs, ACC cholesterol guidelines) |
    | Model unavailable on first run | `predict_health()` trains the model on-the-fly if the pickle file is missing |
    | Streamlit re-runs the entire script on every interaction | Used `st.rerun()` only after mutations; form `clear_on_submit=True` to reset state cleanly |
    | Email uniqueness constraint | Caught `sqlite3.IntegrityError` and surfaced a user-friendly message |
    | Date widget returns a `date` object, DB expects a string | Converted with `str(dob)` → ISO format `YYYY-MM-DD` |
    """)

    st.markdown("---")
    st.caption("MIRA — Medical Intelligence Robotic Automation  •  Built for Gokul Infocare Technical Assessment")


# ─── Sidebar Navigation ───────────────────────────────────────────────────────

def main() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 12px 0 8px;">
                <span style="font-size: 2.4rem;">🏥</span>
                <h2 style="color:#f1f5f9; margin: 4px 0 0; font-size: 1.2rem; letter-spacing: 1px;">MIRA</h2>
                <p style="color:#94a3b8; font-size: 0.72rem; margin: 0;">Medical Intelligence Robotic Automation</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=["🏠 Dashboard", "➕ Add Patient", "📋 Patient Records", "ℹ️ About"],
            label_visibility="collapsed",
        )

        # Quick stats in sidebar footer
        stats = db.get_stats()
        st.markdown("---")
        st.markdown(
            f"""
            <div style="color:#94a3b8; font-size: 0.78rem; text-align: center; line-height: 1.8;">
                👥 {stats['total_patients']} patients &nbsp;|&nbsp;
                ✅ {stats['healthy']} healthy<br>
                ⚠️ {stats['at_risk']} at risk
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Route ─────────────────────────────────────────────────────────────
    if   page == "🏠 Dashboard":      page_dashboard()
    elif page == "➕ Add Patient":     page_add_patient()
    elif page == "📋 Patient Records": page_patient_records()
    elif page == "ℹ️ About":           page_about()


if __name__ == "__main__":
    main()
