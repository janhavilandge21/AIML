"""
database.py — SQLite persistence layer for MIRA Health Prediction App
----------------------------------------------------------------------
Handles all database operations: initialisation, CRUD, and search.
Uses the standard-library sqlite3 module so no extra dependencies are needed.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional

# ─── Configuration ──────────────────────────────────────────────────────────

DB_PATH = "mira_health.db"   # SQLite file lives next to the app


# ─── Connection helper ───────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """
    Return a sqlite3 connection with row_factory set so every row
    behaves like a dict (column-name access instead of index).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows accessible as dicts
    return conn


# ─── Schema initialisation ───────────────────────────────────────────────────

def init_db() -> None:
    """
    Create the patients table if it does not already exist.
    Called once at application startup.

    Schema
    ------
    id          INTEGER  Primary key, auto-increment
    full_name   TEXT     Patient's full name
    dob         TEXT     Date of birth  (ISO format: YYYY-MM-DD)
    email       TEXT     Unique email address
    glucose     REAL     Blood glucose level (mg/dL)
    haemoglobin REAL     Haemoglobin level (g/dL)
    cholesterol REAL     Total cholesterol (mg/dL)
    remarks     TEXT     AI-generated health prediction
    created_at  TEXT     Record creation timestamp
    updated_at  TEXT     Last update timestamp
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name   TEXT    NOT NULL,
            dob         TEXT    NOT NULL,
            email       TEXT    NOT NULL UNIQUE,
            glucose     REAL    NOT NULL,
            haemoglobin REAL    NOT NULL,
            cholesterol REAL    NOT NULL,
            remarks     TEXT    DEFAULT '',
            created_at  TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ─── Create ──────────────────────────────────────────────────────────────────

def create_patient(
    full_name: str,
    dob: str,
    email: str,
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
    remarks: str = ""
) -> tuple[bool, str]:
    """
    Insert a new patient record.

    Returns
    -------
    (True, "Patient added successfully.")  on success
    (False, <error message>)               on failure
    """
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    try:
        conn = get_connection()
        conn.execute(
            """
            INSERT INTO patients
                (full_name, dob, email, glucose, haemoglobin, cholesterol, remarks, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (full_name, dob, email, glucose, haemoglobin, cholesterol, remarks, now, now),
        )
        conn.commit()
        conn.close()
        return True, "Patient record created successfully."
    except sqlite3.IntegrityError:
        return False, f"A patient with the email '{email}' already exists."
    except Exception as exc:
        return False, f"Database error: {exc}"


# ─── Read (all / by id / search) ────────────────────────────────────────────

def get_all_patients() -> list[dict]:
    """Return all patient records as a list of dicts, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM patients ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_patient_by_id(patient_id: int) -> Optional[dict]:
    """Return a single patient dict or None if not found."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM patients WHERE id = ?", (patient_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def search_patients(query: str) -> list[dict]:
    """
    Case-insensitive search across full_name and email columns.
    Returns matching rows as a list of dicts.
    """
    like = f"%{query.strip()}%"
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT * FROM patients
        WHERE full_name LIKE ? OR email LIKE ?
        ORDER BY created_at DESC
        """,
        (like, like),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ─── Update ──────────────────────────────────────────────────────────────────

def update_patient(
    patient_id: int,
    full_name: str,
    dob: str,
    email: str,
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
    remarks: str = ""
) -> tuple[bool, str]:
    """
    Update an existing patient record.

    Returns
    -------
    (True,  "Updated successfully.")  on success
    (False, <error message>)          on failure
    """
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    try:
        conn = get_connection()
        result = conn.execute(
            """
            UPDATE patients
            SET full_name=?, dob=?, email=?, glucose=?, haemoglobin=?,
                cholesterol=?, remarks=?, updated_at=?
            WHERE id=?
            """,
            (full_name, dob, email, glucose, haemoglobin, cholesterol, remarks, now, patient_id),
        )
        conn.commit()
        conn.close()
        if result.rowcount == 0:
            return False, "Patient not found."
        return True, "Patient record updated successfully."
    except sqlite3.IntegrityError:
        return False, f"Another patient already uses the email '{email}'."
    except Exception as exc:
        return False, f"Database error: {exc}"


# ─── Delete ──────────────────────────────────────────────────────────────────

def delete_patient(patient_id: int) -> tuple[bool, str]:
    """
    Delete a patient record by primary key.

    Returns
    -------
    (True,  "Deleted successfully.")  on success
    (False, <error message>)          on failure
    """
    try:
        conn = get_connection()
        result = conn.execute(
            "DELETE FROM patients WHERE id = ?", (patient_id,)
        )
        conn.commit()
        conn.close()
        if result.rowcount == 0:
            return False, "Patient not found."
        return True, "Patient record deleted successfully."
    except Exception as exc:
        return False, f"Database error: {exc}"


# ─── Quick stats (used on dashboard) ────────────────────────────────────────

def get_stats() -> dict:
    """Return aggregate statistics for the dashboard cards."""
    conn = get_connection()
    total   = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    at_risk = conn.execute(
        "SELECT COUNT(*) FROM patients WHERE remarks NOT LIKE '%Healthy%'"
    ).fetchone()[0]
    conn.close()
    return {
        "total_patients": total,
        "at_risk": at_risk,
        "healthy": total - at_risk,
    }
