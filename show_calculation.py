import pandas as pd

# Load your data
df = pd.read_csv('SEER_MLseer_colon_analysis.csv', header=None, skiprows=1)
df.columns = ['Age_at_diagnosis', 'Sex', 'Year_of_diagnosis', 'Race', 
              'Primary_site', 'Histology', 'Grade', 'Stage', 'Patient_ID']

print("=" * 80)
print("HOW I CALCULATED THE NON-METASTATIC PERCENTAGES")
print("=" * 80)

# Step 1: Create metastatic flag based on Stage column
print("\nSTEP 1: Create metastatic flag from Stage column")
print("-" * 80)
print("Logic: If Stage contains 'Distant' -> metastatic = 1, else = 0")
print("Example from your data:")
print("  Row 1: Stage = 'Regional by both...' -> metastatic = 0")
print("  Row 4: Stage = 'Distant site(s)/node(s) involved' -> metastatic = 1")
print("  Row 17: Stage = 'Distant site(s)/node(s) involved' -> metastatic = 1")

df['metastatic'] = df['Stage'].apply(lambda x: 1 if 'Distant' in str(x) else 0)
df['non_metastatic'] = 1 - df['metastatic']

print(f"\nTotal patients: {len(df):,}")
print(f"Metastatic patients: {df['metastatic'].sum():,}")
print(f"Non-metastatic patients: {df['non_metastatic'].sum():,}")

# Step 2: Group by Grade and calculate percentages
print("\n\nSTEP 2: Group by Grade and calculate non-metastatic rate")
print("-" * 80)
print("\nFormula: Non-Metastatic % = (Non-Metastatic Count / Total Count) × 100")

grade_stats = df.groupby('Grade').agg({
    'non_metastatic': ['count', 'sum', 'mean']
})
grade_stats.columns = ['Total_Patients', 'Non_Metastatic_Count', 'Non_Metastatic_Rate']

# Calculate percentages
grade_stats['Non_Metastatic_%'] = (grade_stats['Non_Metastatic_Rate'] * 100).round(2)
grade_stats['Metastatic_Count'] = grade_stats['Total_Patients'] - grade_stats['Non_Metastatic_Count']
grade_stats['Metastatic_%'] = (100 - grade_stats['Non_Metastatic_%']).round(2)

# Show detailed breakdown for key grades
print("\nDETAILED BREAKDOWN FOR KEY GRADES:")
print("=" * 80)

key_grades = ['Well differentiated; Grade I', 'Undifferentiated; anaplastic; Grade IV']

for grade in key_grades:
    if grade in grade_stats.index:
        row = grade_stats.loc[grade]
        print(f"\n{grade}:")
        print(f"  Total Patients: {row['Total_Patients']:,}")
        print(f"  Non-Metastatic Count: {row['Non_Metastatic_Count']:,}")
        print(f"  Metastatic Count: {row['Metastatic_Count']:,}")
        print(f"  Calculation: ({row['Non_Metastatic_Count']:,} / {row['Total_Patients']:,}) x 100")
        print(f"  Non-Metastatic Rate: {row['Non_Metastatic_Rate']:.4f}")
        print(f"  Non-Metastatic %: {row['Non_Metastatic_%']:.2f}%")
        print(f"  Metastatic %: {row['Metastatic_%']:.2f}%")

print("\n\nALL GRADES SUMMARY:")
print("=" * 80)
display_cols = ['Total_Patients', 'Non_Metastatic_Count', 'Metastatic_Count', 
                'Non_Metastatic_%', 'Metastatic_%']
print(grade_stats[display_cols].sort_values('Non_Metastatic_%', ascending=False).to_string())

print("\n\n" + "=" * 80)
print("ANSWER: The values come from counting patients in your SEER dataset")
print("        grouped by Grade, then calculating what % are non-metastatic")
print("=" * 80)
