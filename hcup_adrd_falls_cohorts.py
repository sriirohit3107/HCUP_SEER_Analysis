"""
HCUP NIS — ADRD + Falls Cohort Analysis
Years: 2016–2022 (add paths as available)

Cohorts (mutually exclusive):
  1. ADRD_FALL_PRIMARY  — primary dx (I10_DX1) is a fall (W00-W19); ADRD anywhere
  2. ADRD_FALL_SECONDARY — ADRD anywhere; fall in DX2-DX40 only (not primary admission)
  3. ADRD_NO_FALL       — ADRD anywhere; NO fall code in any DX field

Exclusions:
  - Age < 50
  - Any cancer ICD-10 code C00–C96 in any DX field

ADRD definition (ICD-10):
  G30.x  Alzheimer's disease
  G31.0x Frontotemporal dementia
  G31.83 Dementia with Lewy bodies
  F01.x  Vascular dementia
  F02.x  Dementia in other diseases
  F03.x  Unspecified dementia
  G23.1  Progressive supranuclear palsy

NOTE on NIS patient identity:
  NIS has NO patient-level ID — KEY_NIS is a discharge key.
  Each row is one hospitalization. Same patient may appear multiple times
  across years but cannot be linked. State this as a study limitation.

Table 1 outputs:
  Demographics: age, sex, race, primary payer, zip income quartile
  Spatial: hospital region, hospital location/teaching status
  Clinical: LOS, discharge disposition, top-10 comorbidities (Elixhauser CM_ flags)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── File paths ─────────────────────────────────────────────────────────────────
FILES = {
    2016: r"D:\HCUP\NIS\NIS Decoded\2016\NIS_2016_Core.ASC",
    2017: r"D:\HCUP\NIS\NIS Decoded\2017\NIS_2017_Core.ASC",
    2018: r"D:\HCUP\NIS\NIS Decoded\2018\NIS_2018_Core.ASC",
    2019: r"D:\HCUP\NIS\NIS Decoded\2019\NIS_2019_Core.ASC",
    2020: r"D:\HCUP\NIS\NIS 2020\NIS_2020_Core.ASC",
    2021: r"D:\HCUP\NIS\NIS 2021\NIS_2021\NIS_2021_Core\NIS_2021_Core.ASC",
    2022: r"D:\HCUP\NIS\NIS 2022\NIS_2022\NIS_2022_Core\NIS_2022_Core.ASC",
}

OUTPUT_CSV = r"c:\Users\sprakash1\Documents\seer_ML\HCUP_ADRD_Falls_Cohorts.csv"
TABLE1_CSV = r"c:\Users\sprakash1\Documents\seer_ML\HCUP_Table1.csv"


# ── Column specs (0-indexed, from SAS @position - 1) ──────────────────────────
# Elixhauser CM_ flags are in the HCUP NIS Core file for 2016-2019.
# Column positions for CM_ flags vary by year — add them here as confirmed
# from the SAS load files. Below are the DX/demographic positions confirmed
# from previous analysis; CM_ positions must be verified from SAS load files.

def get_col_dict(year):
    if year == 2016:
        col_dict = {
            'AGE':          (0,   3),
            'FEMALE':       (38,  40),
            'KEY_NIS':      (400, 410),
            'LOS':          (410, 415),
            'PAY1':         (445, 447),      # Primary payer
            'RACE':         (475, 477),
            'ZIPINC_QRTL':  (493, 495),      # Zip income quartile — verify position
            'YEAR':         (491, 495),
            'DISPUNIFORM':  (30,  33),        # Discharge disposition — verify position
            # Hospital-level variables come from HCUP NIS Hospital file (separate file)
            # HOSP_REGION and HOSP_LOCTEACH must be merged via HOSPID
        }
        # DX1–DX30 (principal dx first): @51 + (i-1)*7
        for i in range(1, 31):
            s = 50 + (i - 1) * 7
            col_dict[f'I10_DX{i}'] = (s, s + 7)
        # External cause fields
        for i, pos in enumerate([260, 267, 274, 281], start=1):
            col_dict[f'I10_ECAUSE{i}'] = (pos, pos + 7)

    elif year == 2017:
        col_dict = {
            'AGE':          (0,   3),
            'FEMALE':       (38,  40),
            'KEY_NIS':      (509, 519),
            'LOS':          (519, 524),
            'PAY1':         (554, 556),      # verify position from SAS load file
            'RACE':         (614, 616),
            'ZIPINC_QRTL':  (628, 630),
            'YEAR':         (630, 634),
            'DISPUNIFORM':  (30,  33),
        }
        for i in range(1, 41):
            s = 50 + (i - 1) * 7
            col_dict[f'I10_DX{i}'] = (s, s + 7)

    elif year == 2018:
        col_dict = {
            'AGE':          (0,   3),
            'FEMALE':       (36,  38),
            'KEY_NIS':      (507, 517),
            'LOS':          (517, 522),
            'PAY1':         (552, 554),
            'RACE':         (610, 612),
            'ZIPINC_QRTL':  (624, 626),
            'YEAR':         (626, 630),
            'DISPUNIFORM':  (28,  31),
        }
        for i in range(1, 41):
            s = 48 + (i - 1) * 7
            col_dict[f'I10_DX{i}'] = (s, s + 7)

    else:  # 2019–2022 (NIS 2019 redesigned format aligns with 2020+)
        col_dict = {
            'AGE':          (0,   3),
            'FEMALE':       (36,  38),
            'KEY_NIS':      (520, 530),
            'LOS':          (530, 535),
            'PAY1':         (565, 567),
            'RACE':         (625, 627),
            'ZIPINC_QRTL':  (639, 641),
            'YEAR':         (641, 645),
            'DISPUNIFORM':  (28,  31),
        }
        for i in range(1, 41):
            s = 54 + (i - 1) * 7
            col_dict[f'I10_DX{i}'] = (s, s + 7)

    return col_dict


# ── Vectorized ICD-10 matching (fast — no Python-level loops) ─────────────────

# Regex patterns applied to a single concatenated string per row
FALL_RE        = r'W0\d|W1[0-9]'        # W00–W19 fall mechanism codes
FALL_INJURY_RE = r'S[0-9]\d|T[0-2]\d|T3[0-4]'  # S00-S99, T00-T34 injury codes (fall result)
PRIOR_FALL_RE  = r'Z9181'               # Z91.81 personal history of falling
ADRD_RE        = r'G30|G310|G3183|F01|F02|F03|G231'
CANCER_RE      = r'C[0-8]\d|C9[0-6]'   # C00–C96

ELIXHAUSER_RE = {
    'Hypertension':          r'I1[01235]',
    'Diabetes':              r'E1[0-3]',
    'Congestive_Heart_Fail': r'I50',
    'Cardiac_Arrhythmia':   r'I4[789]',
    'COPD':                  r'J4[0-4]',
    'Depression':            r'F3[234]|F41',
    'Renal_Failure':         r'N1[789]',
    'Obesity':               r'E66',
    'Osteoporosis':          r'M8[01]',
    'Malnutrition':          r'E4[0-6]',
}


def make_dx_string(chunk, dx_cols):
    """Concatenate all DX columns into one string per row for fast regex search."""
    return chunk[dx_cols].fillna('').apply(' '.join, axis=1)


def vec_match(series, pattern):
    """Vectorized: True if pattern found anywhere in the concatenated DX string."""
    return series.str.contains(pattern, regex=True, na=False)


def compute_comorbidities(dx_str):
    """Return a DataFrame of binary Elixhauser flags using vectorized regex."""
    return pd.DataFrame(
        {name: vec_match(dx_str, pat).astype(int)
         for name, pat in ELIXHAUSER_RE.items()},
        index=dx_str.index,
    )


# ── Main processing loop ───────────────────────────────────────────────────────

all_cohort_records = []

for year, path in FILES.items():
    print(f"\nProcessing {year}...")
    col_dict = get_col_dict(year)
    colspecs  = list(col_dict.values())
    names     = list(col_dict.keys())

    dx_cols    = [c for c in names if c.startswith('I10_DX') or c.startswith('I10_ECAUSE')]
    dx1_col    = 'I10_DX1'
    dx_non1    = [c for c in dx_cols if c != dx1_col]

    counts = {'primary_fall': 0, 'prior_fall': 0, 'no_fall': 0,
              'excluded_age': 0, 'excluded_cancer': 0, 'total_adrd': 0}

    try:
        for chunk in pd.read_fwf(
            path,
            colspecs=colspecs,
            names=names,
            header=None,
            chunksize=100_000,
            dtype=str,
        ):
            # -- Numeric conversions
            chunk['AGE'] = pd.to_numeric(chunk['AGE'], errors='coerce')
            chunk['LOS'] = pd.to_numeric(chunk['LOS'], errors='coerce')

            # -- Build concatenated DX string once per chunk (used for all regex)
            dx_str_full = make_dx_string(chunk, dx_cols)

            # -- Step 1: ADRD anywhere (vectorized)
            adrd_mask = vec_match(dx_str_full, ADRD_RE)
            chunk     = chunk[adrd_mask].copy()
            dx_str_full = dx_str_full[adrd_mask]
            if chunk.empty:
                continue
            counts['total_adrd'] += len(chunk)

            # -- Step 2: Exclude age < 50
            age_mask = chunk['AGE'] >= 50
            counts['excluded_age'] += (~age_mask).sum()
            chunk       = chunk[age_mask]
            dx_str_full = dx_str_full[age_mask]
            if chunk.empty:
                continue

            # -- Step 3: Exclude cancer patients (vectorized)
            cancer_mask = vec_match(dx_str_full, CANCER_RE)
            counts['excluded_cancer'] += cancer_mask.sum()
            chunk       = chunk[~cancer_mask]
            dx_str_full = dx_str_full[~cancer_mask]
            if chunk.empty:
                continue

            # -- Step 4: Fall classification (vectorized)
            dx1_str     = chunk[dx1_col].fillna('')
            dx_non1_str = make_dx_string(chunk, dx_non1)

            # Cohort 1 — primary admission reason is a fall:
            #   (a) fall mechanism code W00-W19 directly in DX1 (rare but valid)
            #   (b) fall-related injury (S/T code) in DX1 AND fall mechanism in any secondary DX
            fall_mech_dx1   = dx1_str.str.contains(FALL_RE, regex=True, na=False)
            injury_dx1      = dx1_str.str.contains(FALL_INJURY_RE, regex=True, na=False)
            fall_in_non1    = vec_match(dx_non1_str, FALL_RE)
            primary_fall    = fall_mech_dx1 | (injury_dx1 & fall_in_non1)

            # Cohort 2 — prior fall history (Z91.81 anywhere), not in Cohort 1
            prior_fall_mask = vec_match(dx_str_full, PRIOR_FALL_RE)
            prior_fall      = (~primary_fall) & prior_fall_mask

            # Cohort 3 — no fall of any kind
            fall_any_mask   = vec_match(dx_str_full, FALL_RE) | prior_fall_mask
            no_fall         = ~fall_any_mask & ~primary_fall

            chunk['COHORT'] = np.where(
                primary_fall, 'ADRD_FALL_PRIMARY',
                np.where(
                    prior_fall, 'ADRD_PRIOR_FALL',
                    np.where(no_fall, 'ADRD_NO_FALL', 'UNCLASSIFIED')
                )
            )

            counts['primary_fall'] += primary_fall.sum()
            counts['prior_fall']   += prior_fall.sum()
            counts['no_fall']        += no_fall.sum()

            # -- Step 5: Compute comorbidities (vectorized)
            cm = compute_comorbidities(dx_str_full)
            chunk = pd.concat([chunk, cm], axis=1)

            # -- Retain key columns for Table 1
            keep = ['KEY_NIS', 'YEAR', 'COHORT', 'AGE', 'FEMALE', 'RACE',
                    'PAY1', 'ZIPINC_QRTL', 'LOS', 'DISPUNIFORM', dx1_col] + \
                   list(ELIXHAUSER_RE.keys())

            keep = [c for c in keep if c in chunk.columns]
            all_cohort_records.append(chunk[keep])

        print(f"  ADRD discharges (before exclusions): {counts['total_adrd']:,}")
        print(f"  Excluded age<50:                     {counts['excluded_age']:,}")
        print(f"  Excluded cancer:                     {counts['excluded_cancer']:,}")
        print(f"  Cohort 1 — ADRD + Fall (Primary Adm):{counts['primary_fall']:,}")
        print(f"  Cohort 2 — ADRD + Prior Fall (Z91.81):{counts['prior_fall']:,}")
        print(f"  Cohort 3 — ADRD, NO Fall:             {counts['no_fall']:,}")

    except FileNotFoundError:
        print(f"  FILE NOT FOUND — skipping {year}")


# ── Append new years to existing CSV (do not overwrite previous runs) ──────────
import os
if not all_cohort_records:
    print("\nNo records found. Check file paths.")
else:
    df_new = pd.concat(all_cohort_records, ignore_index=True)
    write_header = not os.path.exists(OUTPUT_CSV)
    df_new.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
    print(f"\nAppended {len(df_new):,} rows to: {OUTPUT_CSV}")

    # reload full file for Table 1
    df = pd.read_csv(OUTPUT_CSV, low_memory=False)
    print(f"Total rows in file so far: {len(df):,}")
    print("Years in file:", sorted(df['YEAR'].dropna().unique().tolist()))

    # ── Table 1 ───────────────────────────────────────────────────────────────
    print("\n========== TABLE 1 ==========")

    table1_rows = []

    for cohort in ['ADRD_FALL_PRIMARY', 'ADRD_PRIOR_FALL', 'ADRD_NO_FALL']:
        sub = df[df['COHORT'] == cohort]
        n   = len(sub)

        row = {'Cohort': cohort, 'N': n}

        # Demographics
        row['Age_mean']    = round(float(sub['AGE'].mean()), 1)
        row['Age_sd']      = round(float(sub['AGE'].std()),  1)
        row['Female_%']    = round(float((sub['FEMALE'].astype(str).str.strip() == '1').mean()) * 100, 1)

        # Race (HCUP codes: 1=White, 2=Black, 3=Hispanic, 4=Asian/PI, 5=Native Am, 6=Other)
        for race_val, race_label in [('1','White'), ('2','Black'), ('3','Hispanic'),
                                      ('4','Asian'), ('5','Native_Am'), ('6','Other')]:
            row[f'Race_{race_label}_%'] = round(
                float((sub['RACE'].astype(str).str.strip() == race_val).mean()) * 100, 1)

        # Primary payer (1=Medicare, 2=Medicaid, 3=Private, 4=Self-pay, 5=No charge, 6=Other)
        for pay_val, pay_label in [('1','Medicare'), ('2','Medicaid'), ('3','Private'),
                                    ('4','SelfPay'), ('6','Other')]:
            row[f'Pay_{pay_label}_%'] = round(
                float((sub['PAY1'].astype(str).str.strip() == pay_val).mean()) * 100, 1)

        # Zip income quartile (1=lowest, 4=highest)
        for q in ['1', '2', '3', '4']:
            row[f'ZipInc_Q{q}_%'] = round(
                float((sub['ZIPINC_QRTL'].astype(str).str.strip() == q).mean()) * 100, 1)

        # LOS
        row['LOS_mean'] = round(float(sub['LOS'].mean()), 1)
        row['LOS_sd']   = round(float(sub['LOS'].std()),  1)

        # Top-10 comorbidities
        for cm in ELIXHAUSER_RE.keys():
            if cm in sub.columns:
                row[f'{cm}_%'] = round(float(sub[cm].mean()) * 100, 1)

        table1_rows.append(row)

    table1 = pd.DataFrame(table1_rows).T
    table1.columns = table1.iloc[0]
    table1 = table1.iloc[1:]

    print(table1.to_string())
    table1.to_csv(TABLE1_CSV)
    print(f"\nSaved: {TABLE1_CSV}")

    # ── Cohort sizes by year ───────────────────────────────────────────────────
    print("\n========== COHORT SIZES BY YEAR ==========")
    summary = df.groupby(['YEAR', 'COHORT']).size().unstack(fill_value=0)
    print(summary.to_string())