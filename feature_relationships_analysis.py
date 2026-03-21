import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy import stats

# Load data
df = pd.read_csv('SEER_MLseer_colon_analysis.csv', header=None, skiprows=1)
df.columns = ['Age_at_diagnosis', 'Sex', 'Year_of_diagnosis', 'Race', 
              'Primary_site', 'Histology', 'Grade', 'Stage', 'Patient_ID']

print("=" * 80)
print("FEATURE RELATIONSHIP ANALYSIS - CAUSAL & CORRELATIONAL RELATIONSHIPS")
print("=" * 80)

# Create metastatic flag
df['metastatic'] = df['Stage'].apply(lambda x: 1 if 'Distant' in str(x) else 0)

print("\n1. AGE vs GRADE RELATIONSHIP")
print("-" * 80)
print("Question: Do older patients have different tumor grades?")
age_grade = pd.crosstab(df['Age_at_diagnosis'], df['Grade'], margins=True)
print("\nCross-tabulation (showing top age groups):")
print(age_grade.head(10))

# Calculate grade distribution by age
print("\nGrade distribution by Age Group:")
age_groups_sample = df['Age_at_diagnosis'].value_counts().head(5).index
for age in age_groups_sample:
    age_df = df[df['Age_at_diagnosis'] == age]
    grade_dist = age_df['Grade'].value_counts(normalize=True).head(3) * 100
    print(f"\n{age}:")
    for grade, pct in grade_dist.items():
        print(f"  {grade[:40]:40s} {pct:5.2f}%")

# Chi-square test
age_grade_chi = pd.crosstab(df['Age_at_diagnosis'], df['Grade'])
chi2, p_value, dof, expected = chi2_contingency(age_grade_chi)
print(f"\nChi-square test: chi2={chi2:.2f}, p-value={p_value:.2e}")
print(f"Interpretation: {'STRONG RELATIONSHIP' if p_value < 0.001 else 'Weak/No relationship'}")

print("\n\n2. PRIMARY SITE vs GRADE RELATIONSHIP")
print("-" * 80)
print("Question: Do different colon locations have different tumor grades?")
site_grade = pd.crosstab(df['Primary_site'], df['Grade'], margins=True)
print("\nCross-tabulation:")
print(site_grade)

# Calculate grade distribution by site
print("\nGrade distribution by Primary Site:")
for site in df['Primary_site'].value_counts().head(5).index:
    site_df = df[df['Primary_site'] == site]
    grade_dist = site_df['Grade'].value_counts(normalize=True).head(3) * 100
    print(f"\n{site}:")
    for grade, pct in grade_dist.items():
        print(f"  {grade[:40]:40s} {pct:5.2f}%")

# Chi-square test
site_grade_chi = pd.crosstab(df['Primary_site'], df['Grade'])
chi2, p_value, dof, expected = chi2_contingency(site_grade_chi)
print(f"\nChi-square test: chi2={chi2:.2f}, p-value={p_value:.2e}")
print(f"Interpretation: {'STRONG RELATIONSHIP' if p_value < 0.001 else 'Weak/No relationship'}")

print("\n\n3. RACE vs GRADE RELATIONSHIP")
print("-" * 80)
print("Question: Do different races have different tumor grades?")
race_grade = pd.crosstab(df['Race'], df['Grade'], margins=True)
print("\nCross-tabulation:")
print(race_grade)

# Chi-square test
race_grade_chi = pd.crosstab(df['Race'], df['Grade'])
chi2, p_value, dof, expected = chi2_contingency(race_grade_chi)
print(f"\nChi-square test: chi2={chi2:.2f}, p-value={p_value:.2e}")
print(f"Interpretation: {'STRONG RELATIONSHIP' if p_value < 0.001 else 'Weak/No relationship'}")

print("\n\n4. AGE vs PRIMARY SITE RELATIONSHIP")
print("-" * 80)
print("Question: Do different age groups have different colon locations?")
age_site = pd.crosstab(df['Age_at_diagnosis'], df['Primary_site'], margins=True)
print("\nCross-tabulation (showing top age groups):")
print(age_site.head(5))

# Chi-square test
age_site_chi = pd.crosstab(df['Age_at_diagnosis'], df['Primary_site'])
chi2, p_value, dof, expected = chi2_contingency(age_site_chi)
print(f"\nChi-square test: chi2={chi2:.2f}, p-value={p_value:.2e}")
print(f"Interpretation: {'STRONG RELATIONSHIP' if p_value < 0.001 else 'Weak/No relationship'}")

print("\n\n5. HISTOLOGY vs GRADE RELATIONSHIP")
print("-" * 80)
print("Question: Do different histology types have different grades?")
hist_grade = pd.crosstab(df['Histology'], df['Grade'], margins=True)
print("\nTop histology types:")
top_hist = df['Histology'].value_counts().head(5).index
for hist in top_hist:
    hist_df = df[df['Histology'] == hist]
    grade_dist = hist_df['Grade'].value_counts(normalize=True).head(3) * 100
    print(f"\nHistology {hist}:")
    for grade, pct in grade_dist.items():
        print(f"  {grade[:40]:40s} {pct:5.2f}%")

# Chi-square test
hist_grade_chi = pd.crosstab(df['Histology'], df['Grade'])
chi2, p_value, dof, expected = chi2_contingency(hist_grade_chi)
print(f"\nChi-square test: chi2={chi2:.2f}, p-value={p_value:.2e}")
print(f"Interpretation: {'STRONG RELATIONSHIP' if p_value < 0.001 else 'Weak/No relationship'}")

print("\n\n6. YEAR vs GRADE RELATIONSHIP (Temporal Trends)")
print("-" * 80)
print("Question: Has tumor grade distribution changed over time?")
df['Year_bin'] = pd.cut(df['Year_of_diagnosis'], bins=[1999, 2005, 2010, 2015, 2023], 
                        labels=['2000-2005', '2006-2010', '2011-2015', '2016-2022'])
year_grade = pd.crosstab(df['Year_bin'], df['Grade'], margins=True)
print("\nGrade distribution over time:")
print(year_grade)

# Chi-square test
year_grade_chi = pd.crosstab(df['Year_bin'], df['Grade'])
chi2, p_value, dof, expected = chi2_contingency(year_grade_chi)
print(f"\nChi-square test: chi2={chi2:.2f}, p-value={p_value:.2e}")
print(f"Interpretation: {'STRONG TEMPORAL TREND' if p_value < 0.001 else 'No temporal trend'}")

print("\n\n7. SEX vs PRIMARY SITE RELATIONSHIP")
print("-" * 80)
sex_site = pd.crosstab(df['Sex'], df['Primary_site'], margins=True)
print("\nCross-tabulation:")
print(sex_site)

# Chi-square test
sex_site_chi = pd.crosstab(df['Sex'], df['Primary_site'])
chi2, p_value, dof, expected = chi2_contingency(sex_site_chi)
print(f"\nChi-square test: chi2={chi2:.2f}, p-value={p_value:.2e}")
print(f"Interpretation: {'STRONG RELATIONSHIP' if p_value < 0.001 else 'Weak/No relationship'}")

print("\n\n8. CONDITIONAL PROBABILITIES - KEY RELATIONSHIPS")
print("-" * 80)
print("\nA. Grade distribution GIVEN Primary Site:")
print("   (Does site influence grade?)")
for site in df['Primary_site'].value_counts().head(3).index:
    site_df = df[df['Primary_site'] == site]
    well_pct = (site_df['Grade'] == 'Well differentiated; Grade I').mean() * 100
    poor_pct = (site_df['Grade'] == 'Poorly differentiated; Grade III').mean() * 100
    print(f"\n   {site}:")
    print(f"     Well differentiated: {well_pct:.2f}%")
    print(f"     Poorly differentiated: {poor_pct:.2f}%")
    print(f"     Overall: Well={df['Grade'].eq('Well differentiated; Grade I').mean()*100:.2f}%, Poor={df['Grade'].eq('Poorly differentiated; Grade III').mean()*100:.2f}%")

print("\n\nB. Primary Site distribution GIVEN Age:")
print("   (Does age influence tumor location?)")
for age in ['40-44 years', '70-74 years', '85-89 years']:
    if age in df['Age_at_diagnosis'].values:
        age_df = df[df['Age_at_diagnosis'] == age]
        sigmoid_pct = (age_df['Primary_site'] == 'C18.7-Sigmoid colon').mean() * 100
        ascending_pct = (age_df['Primary_site'] == 'C18.2-Ascending colon').mean() * 100
        print(f"\n   {age}:")
        print(f"     Sigmoid colon: {sigmoid_pct:.2f}%")
        print(f"     Ascending colon: {ascending_pct:.2f}%")
        print(f"     Overall: Sigmoid={df['Primary_site'].eq('C18.7-Sigmoid colon').mean()*100:.2f}%, Ascending={df['Primary_site'].eq('C18.2-Ascending colon').mean()*100:.2f}%")

print("\n\n9. SUMMARY OF RELATIONSHIPS")
print("=" * 80)
print("\nPOTENTIAL CAUSAL RELATIONSHIPS:")
print("1. Age -> Grade: Older patients may have different tumor biology")
print("2. Primary Site -> Grade: Different locations may have different tumor types")
print("3. Histology -> Grade: Some histologies are inherently higher grade")
print("4. Race -> Grade: May reflect genetic factors or healthcare access")
print("5. Year -> Grade: Temporal trends in detection/classification")
print("\nThese relationships mean:")
print("- Features are NOT independent")
print("- Some features may be redundant or proxy for others")
print("- Model may learn interactions between features")
print("- Multicollinearity may exist (but less critical for tree-based models)")
