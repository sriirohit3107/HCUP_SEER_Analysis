# ==============================================
# Colon Cancer Metastasis Prediction (Inference)
# ==============================================
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------------------
# 1️⃣ Load encoded training feature schema
# ------------------------------------------------------------
train_path = r"C:\Users\sprakash1\Documents\seer_ML\seer_colon_encoded.csv"
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training encoded file not found at {train_path}")

# low_memory=False prevents DtypeWarning on mixed-type columns in the encoded CSV
df_train = pd.read_csv(train_path, low_memory=False)
X_train = df_train.drop(columns=["metastatic"])
print(f"✅ Training schema loaded. Columns: {len(X_train.columns)}")

# ------------------------------------------------------------
# 2️⃣ Load trained model and scaler
# ------------------------------------------------------------
model_path = r"C:\Users\sprakash1\Documents\seer_ML\xgb_model.pkl"
scaler_path = r"C:\Users\sprakash1\Documents\seer_ML\scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("❌ Missing trained model or scaler. Please re-run seer_model.py first.")

xgb_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("✅ Loaded calibrated XGBoost model and scaler.")

# ------------------------------------------------------------
# 3️⃣ Load new patient(s) data
# ------------------------------------------------------------
new_path = r"C:\Users\sprakash1\Documents\seer_ML\new_patients.csv"
if not os.path.exists(new_path):
    raise FileNotFoundError(f"❌ new_patients.csv not found at {new_path}")

# try normal comma separator first, fallback to semicolon
try:
    new_df = pd.read_csv(new_path, quotechar='"', skipinitialspace=True)
except Exception:
    print("⚠️ Retrying with semicolon separator...")
    new_df = pd.read_csv(new_path, sep=';', quotechar='"', skipinitialspace=True)

print(f"✅ Loaded new patient data: {new_df.shape[0]} records")

# Warn about Grade IV or immune-phenotype patients — model was not trained on these
excluded_grades = [
    'Undifferentiated; anaplastic; Grade IV',
    'B-cell; pre-B; B-precursor',
    'T-cell',
    'NK cell; natural killer cell (1995+)',
    'Null cell; non T-non B'
]
if 'Grade' in new_df.columns:
    bad_grade_rows = new_df[new_df['Grade'].isin(excluded_grades)]
    if not bad_grade_rows.empty:
        print(f"\n⚠️ WARNING: {len(bad_grade_rows)} patient(s) have a Grade value the model was NOT trained on:")
        print(bad_grade_rows[['Grade']].to_string())
        print("   These patients will receive unreliable predictions (defaulting toward ~1% probability).")
        print("   Consider excluding them or reviewing their records.")

# ------------------------------------------------------------
# 4️⃣ One-hot encode and align with training columns
# ------------------------------------------------------------
# WHY drop_first=False here (not drop_first=True):
#   During training, pd.get_dummies(drop_first=True) dropped the FIRST category
#   for each feature (e.g., "01-04 years" for Age, "Female" for Sex).
#   If we apply drop_first=True again on only 15 patients, the "first" category
#   found in those 15 rows may be completely different (e.g., "40-44 years"),
#   so that column gets dropped instead — causing a column mismatch and filling
#   real patients with zeros, which collapses all predictions to ~1%.
#
#   With drop_first=False we get ALL dummy columns for the new patients.
#   We then use reindex() to keep exactly the training columns.
#   The reference-category columns (the ones dropped in training) are simply
#   not in X_train.columns, so they get excluded automatically.
#   A patient who belongs to the reference category will correctly have 0s
#   across all dummies for that feature — exactly what training expects.
new_encoded = pd.get_dummies(new_df, drop_first=False)

# Reindex to exactly the training schema:
#   - Columns in training but not in new data → filled with 0 (correct default)
#   - Columns in new data but not in training → dropped automatically
new_encoded = new_encoded.reindex(columns=X_train.columns, fill_value=0)

added_zeros = [c for c in X_train.columns if c not in pd.get_dummies(new_df, drop_first=False).columns]
print(f"✅ Feature alignment complete: {len(X_train.columns)} features aligned.")
if added_zeros:
    print(f"   {len(added_zeros)} columns not present in new patients → set to 0 (reference category baseline).")

# ------------------------------------------------------------
# 5️⃣ Predict metastasis probability
# ------------------------------------------------------------
# WHY we do NOT apply the scaler here:
#   The scaler (StandardScaler) was fitted during training but was only
#   used for Logistic Regression — which needs feature scaling.
#   XGBoost was trained directly on unscaled X_train (raw 0/1 dummy values
#   and actual Year/Histology values). Tree-based models split on thresholds
#   (e.g. "if Year > 2015, go right"), so scaling is irrelevant to them.
#
#   If we apply the scaler here, Year 2017 becomes ~367 and Histology 8140
#   becomes ~20+, both far outside the ranges XGBoost saw during training.
#   This causes every decision tree to route ALL patients to the same leaf,
#   collapsing all predictions to a constant ~1.2% regardless of patient data.
#
#   Fix: pass new_encoded (unscaled) directly to XGBoost.
new_df["Metastasis_Probability"] = xgb_model.predict_proba(new_encoded)[:, 1]
new_df["Prediction"] = np.where(
    new_df["Metastasis_Probability"] > 0.5, "Metastatic", "Non-Metastatic"
)

# ------------------------------------------------------------
# 7️⃣ Assign human-readable risk levels
# ------------------------------------------------------------
def risk_level(p):
    if p >= 0.7:
        return "High risk"
    elif p >= 0.4:
        return "Moderate risk"
    else:
        return "Low risk"

new_df["Risk_Level"] = new_df["Metastasis_Probability"].apply(risk_level)

# ------------------------------------------------------------
# 8️⃣ Save results
# ------------------------------------------------------------
out_path = r"C:\Users\sprakash1\Documents\seer_ML\predicted_results.csv"
new_df.to_csv(out_path, index=False)
print(f"\n🔮 Prediction complete! {len(new_df)} patients processed.\n")
print(new_df[["Metastasis_Probability", "Prediction", "Risk_Level"]])

print(f"\n✅ Results saved to: {out_path}")

# ------------------------------------------------------------
# 9️⃣ Save expected feature list for debugging alignment
# ------------------------------------------------------------
feature_list_path = r"C:\Users\sprakash1\Documents\seer_ML\expected_features.txt"
pd.Series(X_train.columns).to_csv(feature_list_path, index=False, header=False)
print(f"🧾 Expected feature list saved to: {feature_list_path}")

if added_zeros:
    print("\n   Note: zero-filled columns are expected reference-category baselines, not data errors.")
else:
    print("\n🎯 Perfect feature match! Predictions are reliable.")
