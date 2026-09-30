"""
SEER Colon Cancer — Initial Analysis by Stage
"""

import pandas as pd

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(
    r"c:\Users\sprakash1\Documents\seer_ML\SEER_MLseer_colon_analysis.csv",
    header=None,
    names=['Age', 'Sex', 'Year', 'Race', 'Primary_Site', 'Histology', 'Grade', 'Stage', 'Patient_ID']
)

print(f"Total patients: {len(df):,}\n")

# ── 1. Stage Distribution ─────────────────────────────────────────────────────
print("=" * 60)
print("1. STAGE DISTRIBUTION")
print("=" * 60)
stage_counts = df['Stage'].value_counts()
stage_pct    = (stage_counts / len(df) * 100).round(2)
stage_df     = pd.DataFrame({'Count': stage_counts, '%': stage_pct})
print(stage_df.to_string())

# ── 2. Sex by Stage ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. SEX BY STAGE")
print("=" * 60)
sex_stage = pd.crosstab(df['Stage'], df['Sex'], margins=True)
print(sex_stage.to_string())

# ── 3. Race by Stage ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. RACE BY STAGE")
print("=" * 60)
race_stage = pd.crosstab(df['Stage'], df['Race'], margins=True)
print(race_stage.to_string())

# ── 4. Grade by Stage ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. GRADE BY STAGE")
print("=" * 60)
grade_stage = pd.crosstab(df['Stage'], df['Grade'], margins=True)
print(grade_stage.to_string())

# ── 5. Age Group by Stage ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. AGE GROUP BY STAGE (top 5 age groups)")
print("=" * 60)
top_ages   = df['Age'].value_counts().head(5).index
age_subset = df[df['Age'].isin(top_ages)]
age_stage  = pd.crosstab(age_subset['Stage'], age_subset['Age'])
print(age_stage.to_string())

# ── 6. Year Trend by Stage ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. YEAR TREND — DISTANT (METASTATIC) CASES")
print("=" * 60)
distant = df[df['Stage'] == 'Distant site(s)/node(s) involved']
print(distant.groupby('Year').size().reset_index(name='Count').to_string(index=False))

# ── 7. Primary Site by Stage ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. TOP PRIMARY SITES BY STAGE")
print("=" * 60)
site_stage = pd.crosstab(df['Primary_Site'], df['Stage'])
site_stage['Total'] = site_stage.sum(axis=1)
print(site_stage.sort_values('Total', ascending=False).head(10).to_string())

# ── Save summary ──────────────────────────────────────────────────────────────
stage_df.to_csv(r"c:\Users\sprakash1\Documents\seer_ML\seer_stage_summary.csv")
print("\n\nSaved: seer_stage_summary.csv")