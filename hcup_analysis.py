"""
HCUP NIS Analysis — Falls (W00-W19) and Alzheimer's (G30.x) counts
Covers: 2016-2022
Column positions confirmed from official SAS load files for each year.
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── File paths ────────────────────────────────────────────────────────────────
FILES = {
    2016: r"D:\HCUP\NIS\NIS Decoded\2016\NIS_2016_Core.ASC",
    2017: r"D:\HCUP\NIS\NIS Decoded\2017\NIS_2017_Core.ASC",
    2018: r"D:\HCUP\NIS\NIS Decoded\2018\NIS_2018_Core.ASC",
    2019: r"D:\HCUP\NIS\NIS Decoded\2019\NIS_2019_Core.ASC",
    2020: r"D:\HCUP\NIS\NIS 2020\NIS_2020_Core.ASC",
    2021: r"D:\HCUP\NIS\NIS 2021\NIS_2021\NIS_2021_Core\NIS_2021_Core.ASC",
    2022: r"D:\HCUP\NIS\NIS 2022\NIS_2022\NIS_2022_Core\NIS_2022_Core.ASC",
}
# ── Column specs (0-indexed start, end-exclusive, from SAS @positions - 1) ───
def get_col_dict(year):

    if year == 2016:
        # LRECL=497 | 30 DX fields + 4 ECAUSE fields
        # Source: SASload_NIS_2016_Core.SAS
        col_dict = {
            'AGE':     (0,   3),
            'FEMALE':  (38,  40),
            'KEY_NIS': (400, 410),
            'LOS':     (410, 415),
            'RACE':    (475, 477),
            'YEAR':    (491, 495),
        }
        for i in range(1, 31):          # I10_DX1–DX30: @51 + (i-1)*7
            s = 50 + (i - 1) * 7
            col_dict[f'I10_DX{i}'] = (s, s + 7)
        # ECAUSE1–4: @261,@268,@275,@282 — also carry W-codes for 2016
        for i, pos in enumerate([260, 267, 274, 281], start=1):
            col_dict[f'I10_ECAUSE{i}'] = (pos, pos + 7)

    elif year == 2017:
        # LRECL=636 | 40 DX fields
        # Source: SASload_NIS_2017_Core.SAS
        col_dict = {
            'AGE':     (0,   3),
            'FEMALE':  (38,  40),
            'KEY_NIS': (509, 519),
            'LOS':     (519, 524),
            'RACE':    (614, 616),
            'YEAR':    (630, 634),
        }
        for i in range(1, 41):          # I10_DX1–DX40: @51 + (i-1)*7
            s = 50 + (i - 1) * 7
            col_dict[f'I10_DX{i}'] = (s, s + 7)

    elif year in (2018, 2019):
        # LRECL=632 | 40 DX fields | no DXVER, no PRVER vs 2017
        # Source: SASload_NIS_2018_Core.SAS (2019 same structure)
        col_dict = {
            'AGE':     (0,   3),
            'FEMALE':  (36,  38),
            'KEY_NIS': (507, 517),
            'LOS':     (517, 522),
            'RACE':    (610, 612),
            'YEAR':    (626, 630),
        }
        for i in range(1, 41):          # I10_DX1–DX40: @49 + (i-1)*7
            s = 48 + (i - 1) * 7
            col_dict[f'I10_DX{i}'] = (s, s + 7)

    else:  # 2020, 2021, 2022
        # LRECL=647 | 40 DX fields
        # Source: SASload_NIS_2020_Core.SAS
        col_dict = {
            'AGE':     (0,   3),
            'FEMALE':  (36,  38),
            'KEY_NIS': (520, 530),
            'LOS':     (530, 535),
            'RACE':    (625, 627),
            'YEAR':    (641, 645),
        }
        for i in range(1, 41):          # I10_DX1–DX40: @55 + (i-1)*7
            s = 54 + (i - 1) * 7
            col_dict[f'I10_DX{i}'] = (s, s + 7)

    return col_dict


# ── ICD-10 filter functions ───────────────────────────────────────────────────
def is_fall(code):
    """W00–W19: fall-related ICD-10 codes"""
    if pd.isna(code):
        return False
    c = str(code).strip()
    if c and c[0] == 'W' and len(c) >= 3:
        try:
            return 0 <= int(c[1:3]) <= 19
        except ValueError:
            pass
    return False


def is_alzheimer(code):
    """G30.x: Alzheimer's disease ICD-10 codes"""
    if pd.isna(code):
        return False
    return str(code).strip().startswith('G30')


# ── Main loop ─────────────────────────────────────────────────────────────────
results = []
fall_sample = None

for year, path in FILES.items():
    print(f"\nProcessing {year} — {path}")
    col_dict = get_col_dict(year)
    colspecs  = list(col_dict.values())
    names     = list(col_dict.keys())
    dx_cols   = [c for c in names if c.startswith('I10_DX') or c.startswith('I10_ECAUSE')]

    fall_count = 0
    alz_count  = 0
    total      = 0

    try:
        for chunk in pd.read_fwf(
            path,
            colspecs=colspecs,
            names=names,
            header=None,
            chunksize=100_000,
            dtype=str,
        ):
            total += len(chunk)

            fall_mask = chunk[dx_cols].apply(
                lambda col: col.map(is_fall)
            ).any(axis=1)

            alz_mask = chunk[dx_cols].apply(
                lambda col: col.map(is_alzheimer)
            ).any(axis=1)

            fall_count += int(fall_mask.sum())
            alz_count  += int(alz_mask.sum())

            if fall_sample is None and fall_mask.any():
                fall_sample = chunk[fall_mask].iloc[0][
                    ['KEY_NIS', 'AGE', 'FEMALE', 'RACE', 'LOS', 'YEAR',
                     'I10_DX1', 'I10_DX2', 'I10_DX3']
                ]

        print(f"  Total discharges : {total:,}")
        print(f"  Fall patients    : {fall_count:,}")
        print(f"  Alzheimer's      : {alz_count:,}")

        results.append({
            'Year':               year,
            'Total_Discharges':   total,
            'Fall_Patients':      fall_count,
            'Alzheimer_Patients': alz_count,
        })

    except FileNotFoundError:
        print(f"  FILE NOT FOUND — skipping {year}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n\n========== SUMMARY ==========")
df = pd.DataFrame(results)
df['Fall_%']      = (df['Fall_Patients']      / df['Total_Discharges'] * 100).round(2)
df['Alzheimer_%'] = (df['Alzheimer_Patients'] / df['Total_Discharges'] * 100).round(2)
print(df.to_string(index=False))

df.to_csv(r"c:\Users\sprakash1\Documents\seer_ML\HCUP_Falls_Alzheimer_Counts.csv", index=False)
print("\nSaved: HCUP_Falls_Alzheimer_Counts.csv")

# ── Sample fall patient record ────────────────────────────────────────────────
if fall_sample is not None:
    print("\n========== SAMPLE FALL PATIENT RECORD ==========")
    print(fall_sample.to_string())