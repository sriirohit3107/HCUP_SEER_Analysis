"""
HCUP ADRD Falls — Table 1 Generator
Reads from: HCUP_ADRD_Falls_Cohorts.csv  (pre-built by hcup_adrd_falls_cohorts.py)
Run this AFTER the cohort CSV is fully built.
"""

import pandas as pd
import os

CSV_PATH  = r"c:\Users\sprakash1\Documents\seer_ML\HCUP_ADRD_Falls_Cohorts.csv"
OUT_TABLE1 = r"c:\Users\sprakash1\Documents\seer_ML\HCUP_Table1.csv"
OUT_YEARLY = r"c:\Users\sprakash1\Documents\seer_ML\HCUP_Cohort_By_Year.csv"

ELIXHAUSER_COLS = [
    'Hypertension', 'Diabetes', 'Congestive_Heart_Fail', 'Cardiac_Arrhythmia',
    'COPD', 'Depression', 'Renal_Failure', 'Obesity', 'Osteoporosis', 'Malnutrition'
]

# ── Load CSV ───────────────────────────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}\nRun hcup_adrd_falls_cohorts.py first.")

print(f"Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH, low_memory=False)
df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
df['LOS'] = pd.to_numeric(df['LOS'], errors='coerce')

print(f"Total rows loaded : {len(df):,}")
print(f"Years in data     : {sorted(df['YEAR'].dropna().unique().astype(int).tolist())}")
print(f"Cohort counts:\n{df['COHORT'].value_counts().to_string()}\n")

# ── Table 1 ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("TABLE 1 — Demographics & Comorbidities by Cohort")
print("=" * 60)

table1_rows = []

for cohort in ['ADRD_FALL_PRIMARY', 'ADRD_PRIOR_FALL', 'ADRD_NO_FALL']:
    sub = df[df['COHORT'] == cohort].copy()
    n   = len(sub)
    if n == 0:
        print(f"  WARNING: {cohort} has 0 records — skipping")
        continue

    row = {'Cohort': cohort, 'N': f"{n:,}"}

    # Age
    row['Age_mean (SD)'] = f"{round(float(sub['AGE'].mean()), 1)} ({round(float(sub['AGE'].std()), 1)})"

    # Sex
    row['Female_%'] = f"{round(float((sub['FEMALE'].astype(str).str.strip() == '1').mean()) * 100, 1)}%"

    # Race (1=White, 2=Black, 3=Hispanic, 4=Asian/PI, 5=Native Am, 6=Other)
    for val, label in [('1','White'), ('2','Black'), ('3','Hispanic'),
                       ('4','Asian/PI'), ('5','Native Am'), ('6','Other')]:
        row[f'Race_{label}_%'] = f"{round(float((sub['RACE'].astype(str).str.strip() == val).mean()) * 100, 1)}%"

    # Primary payer (1=Medicare, 2=Medicaid, 3=Private, 4=Self-pay, 6=Other)
    for val, label in [('1','Medicare'), ('2','Medicaid'), ('3','Private'),
                       ('4','Self-pay'), ('6','Other')]:
        row[f'Payer_{label}_%'] = f"{round(float((sub['PAY1'].astype(str).str.strip() == val).mean()) * 100, 1)}%"

    # Zip income quartile
    for q in ['1', '2', '3', '4']:
        row[f'ZipInc_Q{q}_%'] = f"{round(float((sub['ZIPINC_QRTL'].astype(str).str.strip() == q).mean()) * 100, 1)}%"

    # LOS
    row['LOS_mean (SD)'] = f"{round(float(sub['LOS'].mean()), 1)} ({round(float(sub['LOS'].std()), 1)})"

    # Discharge disposition
    for val, label in [('1','Routine'), ('2','Short-term hosp'), ('3','SNF'),
                       ('6','Home Health'), ('7','AMA'), ('20','Died')]:
        row[f'Dispo_{label}_%'] = f"{round(float((sub['DISPUNIFORM'].astype(str).str.strip() == val).mean()) * 100, 1)}%"

    # Top-10 comorbidities
    for cm in ELIXHAUSER_COLS:
        if cm in sub.columns:
            row[f'{cm}_%'] = f"{round(float(sub[cm].mean()) * 100, 1)}%"
        else:
            row[f'{cm}_%'] = 'N/A'

    table1_rows.append(row)

table1 = pd.DataFrame(table1_rows).set_index('Cohort').T
print(table1.to_string())
table1.to_csv(OUT_TABLE1)
print(f"\nSaved Table 1: {OUT_TABLE1}")

# ── Cohort sizes by year ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COHORT SIZES BY YEAR")
print("=" * 60)
yearly = df.groupby(['YEAR', 'COHORT']).size().unstack(fill_value=0)
print(yearly.to_string())
yearly.to_csv(OUT_YEARLY)
print(f"\nSaved yearly breakdown: {OUT_YEARLY}")

# ── Quick summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("QUICK SUMMARY")
print("=" * 60)
print(f"Total ADRD admissions (all years, post-exclusion): {len(df):,}")
print(f"Cohort 1 — Fall PRIMARY   : {len(df[df['COHORT']=='ADRD_FALL_PRIMARY']):,}")
print(f"Cohort 2 — Fall SECONDARY : {len(df[df['COHORT']=='ADRD_FALL_SECONDARY']):,}")
print(f"Cohort 3 — No Fall        : {len(df[df['COHORT']=='ADRD_NO_FALL']):,}")
