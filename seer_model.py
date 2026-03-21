# ================================
# Colon Cancer Metastasis Prediction using SEER data
# ================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

import joblib

# 1️⃣ Load the SEER data (update path if needed)
path = r"C:\Users\sprakash1\Documents\seer_ML\SEER_MLseer_colon_analysis.csv"
df = pd.read_csv(path, header=None, skiprows=1)

# 2️⃣ Assign proper column names (since SEER exported without headers)
df.columns = [
    'Age_at_diagnosis',
    'Sex',
    'Year_of_diagnosis',
    'Race',
    'Primary_site',
    'Histology',
    'Grade',
    'Stage',
    'Patient_ID'
]

print("✅ Data loaded successfully:", df.shape)
print(df.head())

# ============================================================
# FILTER: Keep only Grade I, II, III
# ============================================================
# WHY WE DO THIS:
#   Our goal is to model cure rate and longevity for the standard
#   colon cancer population (well, moderately, and poorly differentiated
#   tumors). Two categories of rows must be excluded:
#
#   1. GRADE IV (Undifferentiated/Anaplastic) — ~9,999 patients
#      These tumors are biologically so different from Grade I-III
#      that including them would distort the model. Grade IV has
#      dramatically worse outcomes and should eventually be modeled
#      separately. Including them skews predictions for the Grade I-III
#      population we are focused on.
#
#   2. B-cell, T-cell, NK cell, Null cell — ~3,670 patients
#      These are immunophenotype classifications (immune cell lineage
#      labels) that appear in the Grade column due to how SEER encodes
#      multi-cancer data. Colon cancer does not have B-cell or T-cell
#      phenotypes — these are data artifacts or non-colon primaries.
#      Keeping them creates meaningless one-hot columns (Grade_B-cell)
#      that add noise and no predictive signal.
#
# HOW WE DO IT:
#   We define a list of the three valid Grade values and use
#   df[df['Grade'].isin(valid_grades)] to keep only matching rows.
#   One filter handles both exclusions at once.
#   Blank(s) and Unknown are intentionally kept — they represent
#   missing clinical data, which is real and informative.
# ============================================================
valid_grades = [
    'Well differentiated; Grade I',
    'Moderately differentiated; Grade II',
    'Poorly differentiated; Grade III',
    'Blank(s)',
    'Unknown'
]
df = df[df['Grade'].isin(valid_grades)]
print(f"\n✅ After grade filtering: {df.shape[0]:,} patients retained")
print("   (Removed: Grade IV, B-cell, T-cell, NK cell, Null cell)")
print(df['Grade'].value_counts())

# 3️⃣ Create binary target: metastatic = 1 if 'Distant' in Stage, else 0
df['metastatic'] = df['Stage'].apply(lambda x: 1 if 'Distant' in str(x) else 0)
print("\nMetastatic distribution:\n", df['metastatic'].value_counts())

# 4️⃣ Select relevant features
features = [
    'Age_at_diagnosis',
    'Sex',
    'Year_of_diagnosis',
    'Race',
    'Primary_site',
    'Histology',
    'Grade'
]

X = df[features].copy()
y = df['metastatic']

# 5️⃣ One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# 6️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 7️⃣ Scale numeric features (for logistic regression)
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8️⃣ Logistic Regression (baseline model)
# WHY class_weight='balanced': The dataset is heavily imbalanced (~83% non-metastatic,
# ~17% metastatic). Without balancing, LR ignores the minority class entirely and
# predicts non-metastatic for everything — giving recall=0.00 for metastatic patients,
# which is clinically useless. Balanced weighting penalizes misclassifying the minority
# class more heavily, forcing the model to actually learn metastatic patterns.
# WHY solver='saga': Scales better to large datasets than the default 'lbfgs'.
# WHY max_iter=2000: The default 1000 was not enough for convergence on this dataset size.
log_reg = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight='balanced',
    solver='saga'
)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n📊 Logistic Regression Results:")
print("AUC-ROC:", round(roc_auc_score(y_test, y_pred_lr), 3))
print(classification_report(y_test, (y_pred_lr > 0.5).astype(int)))

# 9️⃣ XGBoost (balanced & calibrated model)
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train)  # balance classes
)
xgb_calibrated = CalibratedClassifierCV(xgb, method='sigmoid', cv=3)

xgb_calibrated.fit(X_train, y_train)
y_pred_xgb = xgb_calibrated.predict_proba(X_test)[:, 1]

print("\n📈 XGBoost (Calibrated) Results:")
print("AUC-ROC:", round(roc_auc_score(y_test, y_pred_xgb), 3))
print(classification_report(y_test, (y_pred_xgb > 0.5).astype(int)))

# 🔟 Save encoded dataset for later use
output_path = r"C:\Users\sprakash1\Documents\seer_ML\seer_colon_encoded.csv"
df_model = pd.concat([pd.DataFrame(X), y.reset_index(drop=True)], axis=1)
df_model.to_csv(output_path, index=False)
print(f"\n✅ Encoded data saved to: {output_path}")

# 💾 Save trained model and scaler for inference
joblib.dump(xgb_calibrated, r"C:\Users\sprakash1\Documents\seer_ML\xgb_model.pkl")
joblib.dump(scaler, r"C:\Users\sprakash1\Documents\seer_ML\scaler.pkl")
print("💾 Trained model and scaler saved successfully for future predictions!")