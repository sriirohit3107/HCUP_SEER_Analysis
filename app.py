# ==============================================
# Colon Cancer Metastasis Risk - Doctor Web App
# ==============================================
# Run: streamlit run app.py
# Doctors enter age, sex, grade, primary site, histology (and optionally year, race)
# and get: non-metastatic probability + risk level (Low / Moderate / High)
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Paths (same folder as app)
BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE, "seer_colon_encoded.csv")
MODEL_PKL = os.path.join(BASE, "xgb_model.pkl")
SCALER_PKL = os.path.join(BASE, "scaler.pkl")

# Dropdown options (must match SEER labels exactly)
AGE_OPTIONS = [
    "01-04 years", "05-09 years", "10-14 years", "15-19 years", "20-24 years",
    "25-29 years", "30-34 years", "35-39 years", "40-44 years", "45-49 years",
    "50-54 years", "55-59 years", "60-64 years", "65-69 years", "70-74 years",
    "75-79 years", "80-84 years", "85-89 years", "90+ years",
]
SEX_OPTIONS = ["Female", "Male"]
RACE_OPTIONS = [
    "American Indian/Alaska Native", "Asian or Pacific Islander", "Black",
    "Unknown", "White",
]
PRIMARY_SITE_OPTIONS = [
    "C18.1-Appendix", "C18.2-Ascending colon", "C18.3-Hepatic flexure of colon",
    "C18.4-Transverse colon", "C18.5-Splenic flexure of colon", "C18.6-Descending colon",
    "C18.7-Sigmoid colon", "C18.8-Overlapping lesion of colon", "C18.9-Colon, NOS",
]
# GRADE OPTIONS — intentionally limited to Grade I, II, III only.
# Grade IV (Undifferentiated/Anaplastic) and immune phenotype categories
# (B-cell, T-cell, NK cell, Null cell) are excluded here because the
# model was trained without them. Presenting options the model has never
# seen would produce unreliable predictions (the missing columns would
# default to 0, silently distorting the output).
GRADE_OPTIONS = [
    "Blank(s)",
    "Unknown",
    "Well differentiated; Grade I",
    "Moderately differentiated; Grade II",
    "Poorly differentiated; Grade III",
]
# Common histology codes (ICD-O-3) - user can type others
HISTOLOGY_DEFAULT = [8140, 8260, 8261, 8263, 8480, 8481, 8210, 8220, 8000, 8010, 8020, 8041]

def risk_level(p_metastatic):
    if p_metastatic >= 0.7:
        return "High risk"
    elif p_metastatic >= 0.4:
        return "Moderate risk"
    else:
        return "Low risk"

@st.cache_resource
def load_model_and_schema():
    if not os.path.exists(MODEL_PKL) or not os.path.exists(SCALER_PKL):
        return None, None, None
    if not os.path.exists(TRAIN_CSV):
        return None, None, None
    df_train = pd.read_csv(TRAIN_CSV, low_memory=False)
    X_train = df_train.drop(columns=["metastatic"])
    model = joblib.load(MODEL_PKL)
    scaler = joblib.load(SCALER_PKL)
    return model, scaler, X_train.columns.tolist()

def predict_one_row(row_df, model, scaler, train_columns):
    # Use drop_first=False so all dummy columns are created, then reindex to
    # exactly match the training schema. This avoids the reference-category
    # mismatch bug where drop_first=True on a single row drops the wrong column.
    new_encoded = pd.get_dummies(row_df, drop_first=False)
    new_encoded = new_encoded.reindex(columns=train_columns, fill_value=0)
    # Do NOT apply scaler — XGBoost was trained on unscaled data.
    # Scaling would push Year/Histology far outside training ranges, collapsing
    # all predictions to ~1.2% (same leaf for every patient).
    p_metastatic = model.predict_proba(new_encoded.values)[0, 1]
    return float(p_metastatic)

def main():
    st.set_page_config(page_title="Colon Cancer Metastasis Risk", layout="centered")
    st.title("Colon Cancer Metastasis Risk Tool")
    st.caption("Enter patient demographics and tumor characteristics to estimate probability of being non-metastatic (risk stratification only; not a substitute for staging).")

    model, scaler, train_columns = load_model_and_schema()
    if model is None:
        st.error(
            "Model not found. Please run **seer_model.py** first to train and save "
            "**xgb_model.pkl** and **scaler.pkl** in this folder, and ensure **seer_colon_encoded.csv** exists."
        )
        return

    with st.form("patient_form"):
        st.subheader("Patient & tumor inputs")
        c1, c2 = st.columns(2)
        with c1:
            age = st.selectbox("Age at diagnosis", AGE_OPTIONS, index=AGE_OPTIONS.index("55-59 years") if "55-59 years" in AGE_OPTIONS else 0)
            sex = st.selectbox("Sex", SEX_OPTIONS)
            year = st.number_input("Year of diagnosis", min_value=2000, max_value=2030, value=2022)
        with c2:
            race = st.selectbox("Race", RACE_OPTIONS, index=4)
            primary_site = st.selectbox("Primary site", PRIMARY_SITE_OPTIONS)
            histology = st.number_input("Histology (ICD-O-3 code)", min_value=8000, max_value=9999, value=8140, step=1)
        grade = st.selectbox("Grade", GRADE_OPTIONS, index=GRADE_OPTIONS.index("Moderately differentiated; Grade II") if "Moderately differentiated; Grade II" in GRADE_OPTIONS else 0)
        submitted = st.form_submit_button("Get risk estimate")

    if not submitted:
        st.info("Fill the form and click **Get risk estimate**.")
        return

    row = pd.DataFrame([{
        "Age_at_diagnosis": age,
        "Sex": sex,
        "Year_of_diagnosis": year,
        "Race": race,
        "Primary_site": primary_site,
        "Histology": histology,
        "Grade": grade,
    }])

    try:
        p_metastatic = predict_one_row(row, model, scaler, train_columns)
    except Exception as e:
        st.exception(e)
        return

    p_non_metastatic = 1.0 - p_metastatic
    level = risk_level(p_metastatic)

    st.subheader("Result")
    st.metric("Probability of being NON-metastatic", f"{p_non_metastatic * 100:.1f}%")
    st.metric("Probability of metastasis", f"{p_metastatic * 100:.1f}%")
    if level == "Low risk":
        st.success(f"Metastasis risk level: **{level}**")
    elif level == "Moderate risk":
        st.warning(f"Metastasis risk level: **{level}**")
    else:
        st.error(f"Metastasis risk level: **{level}**")

    st.caption(
        "This tool is for risk stratification only. It does not replace clinical staging. "
        "Metastasis must be determined by imaging and pathology."
    )

if __name__ == "__main__":
    main()
