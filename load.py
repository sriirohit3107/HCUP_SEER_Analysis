import pandas as pd
import os

# Path to your SEER export file
path = r"C:\Users\sprakash1\Documents\seer_ML\export.txt"

# Check if file exists before loading
if not os.path.exists(path):
    raise FileNotFoundError(f"File not found: {path}")

# Load SEER export (tab-delimited)
df = pd.read_csv(path, sep="\t", low_memory=False)

# Preview first few rows
print(df.head())

# Save as CSV for future use
out_path = r"C:\Users\sprakash1\Documents\seer_ML\SEER_MLseer_colon_analysis.csv"
df.to_csv(out_path, index=False)
print(f"\n✅ Saved cleaned CSV to: {out_path}")
