# Seer ML — Colon Cancer Cure Rate & Longevity Prediction

---

## Overall Project Objective

The goal of this project is to build a machine learning system that can predict **cure rate and longevity** for colon cancer patients. We use the **SEER (Surveillance, Epidemiology, and End Results)** database — a large US population-level cancer registry with 800,000+ colon cancer patients — as our training foundation.

The immediate output is a **metastasis risk score** (probability that a patient is non-metastatic at diagnosis). This is the first step toward cure rate and longevity modeling, because:

- Non-metastatic patients have vastly better survival outcomes
- Metastasis status at diagnosis is the single strongest predictor of longevity
- A calibrated risk score lets clinicians stratify patients before full staging workup

**Planned next step:** Validate the model on the **Guthrie dataset** (real clinical data), which will have richer inputs (labs, treatment, CEA levels, etc.) that SEER does not have — enabling a more complete cure rate / longevity model.

---

## What the Model Currently Predicts

| Output | Description |
|---|---|
| Metastasis probability | 0–1 score: how likely is distant metastasis at diagnosis |
| Prediction label | Metastatic / Non-Metastatic (threshold: 0.5) |
| Risk level | Low / Moderate / High risk |

The model is trained on 7 input features available at initial diagnosis. It is **not a diagnostic tool** — it is a risk stratification aid.

---

## Dataset: SEER Colon Cancer Export

**Source:** SEER*Stat export — US National Cancer Institute registry

**File:** `SEER_MLseer_colon_analysis.csv`
**Size:** ~823,000 patients
**Columns (9 total):**

| Column | Description | Example |
|---|---|---|
| Age_at_diagnosis | Age group at diagnosis | "60-64 years" |
| Sex | Patient sex | "Male" / "Female" |
| Year_of_diagnosis | Year cancer was diagnosed | 2008 |
| Race | Race/ethnicity category | "White", "Black" |
| Primary_site | ICD-O site code for colon location | "C18.7-Sigmoid colon" |
| Histology | ICD-O-3 histology code (tumor cell type) | 8140 (adenocarcinoma) |
| Grade | Tumor differentiation grade | "Moderately differentiated; Grade II" |
| Stage | SEER summary stage | "Localized only", "Distant site(s)..." |
| Patient_ID | Anonymized patient ID | 00000377 |

**Grade values in data (before filtering):**

| Grade Value | Count | Keep? |
|---|---|---|
| Moderately differentiated; Grade II | 358,423 | YES |
| Blank(s) | 180,637 | YES (unknown — kept) |
| Unknown | 115,200 | YES (unknown — kept) |
| Poorly differentiated; Grade III | 94,987 | YES |
| Well differentiated; Grade I | 60,709 | YES |
| Undifferentiated; anaplastic; Grade IV | 9,999 | **NO — excluded** |
| B-cell; pre-B; B-precursor | 3,545 | **NO — excluded** |
| T-cell | 114 | **NO — excluded** |
| NK cell; natural killer cell (1995+) | 7 | **NO — excluded** |
| Null cell; non T-non B | 4 | **NO — excluded** |

---

## Why We Exclude Grade IV and B-cell/Immune Categories

### Item 1: Exclude Grade IV (Undifferentiated/Anaplastic)

**What:** Rows where `Grade = "Undifferentiated; anaplastic; Grade IV"` (~10,000 patients).

**Why:**
- Grade IV represents a biologically distinct disease from Grade I–III
- These tumors are so poorly differentiated that they behave almost like a different cancer
- Including them would teach the model to generalize across a range it should not — the biology does not form a continuous spectrum from Grade I to IV in the same way
- For cure rate and longevity modeling, Grade IV patients have dramatically different outcomes that would skew predictions for the Grade I–III population we are focused on

**How:** Filter rows to only keep the three valid differentiation grades before any encoding or model training.

### Item 2: Exclude B-cell, T-cell, NK cell, Null cell (Immune/Lymphoma Classifications)

**What:** Rows where Grade is one of: `B-cell; pre-B; B-precursor`, `T-cell`, `NK cell; natural killer cell (1995+)`, `Null cell; non T-non B` (~3,670 patients total).

**Why:**
- These values are **immunophenotype classifications** — they describe immune cell lineage, not tumor differentiation
- They exist in the Grade column because the SEER data dictionary uses the Grade field for multiple purposes depending on cancer type
- For colon cancer, these values are data entry artifacts — colon cancer does not have B-cell or T-cell phenotypes; these records likely represent non-colon primaries or data errors
- Keeping them would create nonsensical one-hot encoded columns (`Grade_B-cell`) that carry no predictive signal and pollute the model's feature space

**How:** The same `valid_grades` filter (Step 1 above) handles this automatically — by specifying only the three valid grades, everything else (including all immune categories) is excluded.

---

## File-by-File Explanation

### Data Files

#### `export.txt`
Raw SEER export file in tab-delimited format. This is the original data file downloaded directly from SEER*Stat. It is the source of truth. All other data files are derived from this.

#### `SEER_MLseer_colon_analysis.csv`
The main working dataset. Generated from `export.txt` by `load.py`. This is what all training and analysis scripts read. Contains all 823,000+ patients with 9 columns.

#### `export.dic`
SEER data dictionary file that accompanies the export. Describes the variable codes and their meanings (e.g., what histology code 8140 means). Reference file — not read by any Python script.

#### `seer_colon_encoded.csv`
The one-hot encoded version of the training dataset. Generated by `seer_model.py` after training. Contains all the expanded dummy columns (e.g., `Grade_Moderately differentiated; Grade II`, `Sex_Male`, etc.) plus the `metastatic` target column. Used by `seer_predict.py` and `app.py` to know what columns the model expects.

#### `new_patients.csv`
Input file for running predictions on new patients. You populate this with new patient records in the same column format as the training data (Age_at_diagnosis, Sex, Year_of_diagnosis, Race, Primary_site, Histology, Grade). Run `seer_predict.py` to get predictions.

#### `predicted_results.csv`
Output file from `seer_predict.py`. Contains the original new patient data plus three added columns: `Metastasis_Probability` (0–1 score), `Prediction` (Metastatic/Non-Metastatic), and `Risk_Level` (Low/Moderate/High).

#### `expected_features.txt`
A plain-text list of all feature column names the trained model expects (one per line). Generated automatically by `seer_predict.py`. Used for debugging when a new patient's data has mismatched column names. If a feature is missing in new data, it is filled with 0.

---

### Model Files

#### `xgb_model.pkl`
The trained and calibrated XGBoost classification model. Saved by `seer_model.py`. Loaded by `seer_predict.py` and `app.py` for inference. Must be regenerated by rerunning `seer_model.py` any time the training data or feature set changes.

**Model type:** XGBoost → probability-calibrated via Platt scaling (sigmoid calibration, 3-fold CV)
**Why calibrated:** Raw XGBoost probabilities can be overconfident. Calibration ensures the 0.7 probability output actually means ~70% of those patients are metastatic.

#### `scaler.pkl`
The fitted `StandardScaler` from scikit-learn. Applied to numeric features before model input. Saved and loaded alongside `xgb_model.pkl`. Must match the training data it was fitted on — regenerate when retraining.

---

### Python Scripts

#### `load.py`
**Purpose:** One-time data loading and conversion
**What it does:** Reads `export.txt` (tab-delimited SEER export), saves it as `SEER_MLseer_colon_analysis.csv`
**Run when:** You have a new SEER export and need to convert it
**Output:** `SEER_MLseer_colon_analysis.csv`

---

#### `seer_model.py`
**Purpose:** Core model training script — the heart of the pipeline
**What it does, step by step:**

1. Loads `SEER_MLseer_colon_analysis.csv`
2. **[Updated]** Filters to only Grade I, II, III — removes Grade IV and all immune phenotype categories
3. Creates binary target: `metastatic = 1` if Stage contains "Distant", else 0
4. Selects 7 features: Age, Sex, Year, Race, Primary_site, Histology, Grade
5. One-hot encodes categorical features (`pd.get_dummies`)
6. Splits 70% train / 30% test (stratified)
7. Scales numeric features with `StandardScaler`
8. Trains Logistic Regression as a baseline
9. Trains XGBoost (class-balanced) → calibrates probabilities with Platt scaling
10. Prints AUC-ROC and classification reports for both models
11. Saves encoded dataset → `seer_colon_encoded.csv`
12. Saves model → `xgb_model.pkl`
13. Saves scaler → `scaler.pkl`

**Run when:** You change the data, change the features, or change filtering logic
**Output:** `seer_colon_encoded.csv`, `xgb_model.pkl`, `scaler.pkl`

---

#### `seer_predict.py`
**Purpose:** Run predictions on new patients using the saved model
**What it does, step by step:**

1. Loads `seer_colon_encoded.csv` to get the expected feature schema
2. Loads `xgb_model.pkl` and `scaler.pkl`
3. Loads `new_patients.csv` (new patients to score)
4. One-hot encodes the new patients
5. Aligns columns to match training schema (adds missing columns as 0, drops extra columns)
6. Scales features and runs prediction
7. Adds `Metastasis_Probability`, `Prediction`, and `Risk_Level` to results
8. Saves to `predicted_results.csv`
9. Saves the expected feature list to `expected_features.txt`

**Run when:** You have new patients and want predictions
**Input:** `new_patients.csv`
**Output:** `predicted_results.csv`, `expected_features.txt`

---

#### `app.py`
**Purpose:** Streamlit web application — doctor-facing UI
**What it does:** Provides a form where a clinician can enter a patient's demographics and tumor characteristics, then get an instant metastasis risk estimate
**How to run:** `streamlit run app.py`
**Inputs (via UI dropdowns):** Age, Sex, Year, Race, Primary site, Histology code, Grade
**Outputs (displayed):** % probability of being non-metastatic, % probability of metastasis, Risk level (Low/Moderate/High)
**Loads:** `xgb_model.pkl`, `scaler.pkl`, `seer_colon_encoded.csv`

**Note:** Grade dropdown in `app.py` currently still lists Grade IV and B-cell options — these need to be removed to match the updated training data.

---

#### `analyze_patterns.py`
**Purpose:** Exploratory analysis — understand non-metastatic rates by feature group
**What it does:** Groups patients by Grade, Primary Site, Sex, and Race, calculates the % non-metastatic for each group, and prints a summary
**Key finding it shows:** Grade is the strongest predictor (91.9% non-metastatic for Grade I vs 75.3% for Grade IV)
**Run when:** You want to understand the data patterns before/after filtering
**Not part of the prediction pipeline** — analysis only

---

#### `feature_relationships_analysis.py`
**Purpose:** Statistical analysis of relationships between input features
**What it does:** Runs chi-square tests on every feature pair to measure how strongly they are correlated. Answers questions like "Do older patients have different tumor grades?" or "Does primary site influence grade?"
**Key findings:**
- Histology → Grade is the strongest relationship (chi2 = 1,654,303)
- Primary site → Grade is also strong (chi2 = 174,063)
- Age → Primary site is significant (older patients get more right-side tumors)
- All feature pairs are statistically significant (p < 0.001)

**Not part of the prediction pipeline** — analysis only

---

#### `show_calculation.py`
**Purpose:** Transparency / explainability tool
**What it does:** Shows exactly how the non-metastatic percentages were calculated — counts patients by Grade, shows the numerator/denominator math, prints a full breakdown table
**Use case:** When someone asks "how did you get 91.9% non-metastatic for Grade I?" — run this script
**Not part of the prediction pipeline** — explanation only

---

### Documentation Files

#### `feature_importance_analysis.md`
Detailed write-up of what each feature contributes to the model's predictions. Ranks features by predictive power (Grade > Primary Site > Histology > Age > Race > Sex > Year). Documents what clinical features are missing that would improve the model (T/N stage, CEA, tumor size, MSI status).

#### `causal_relationships_summary.md`
Results from the chi-square analysis. Documents the causal chain: Age → Primary Site → Histology → Grade → Metastasis Risk. Explains why feature correlation is acceptable for XGBoost (tree models handle interactions natively).

#### `requirements-webapp.txt`
Python package requirements for running the Streamlit web app (`app.py`). Install with `pip install -r requirements-webapp.txt`.

---

## Pipeline Flow

```
export.txt
    |
    v
load.py  -->  SEER_MLseer_colon_analysis.csv
                        |
                        v
                  seer_model.py
                  [Filter: Grade I/II/III only]
                  [Train XGBoost + calibrate]
                        |
              __________|__________
             |           |         |
             v           v         v
   seer_colon_encoded  xgb_model  scaler
        .csv             .pkl      .pkl
             |           |         |
             |___________|_________|
                         |
              ___________|___________
             |                       |
             v                       v
       seer_predict.py           app.py
       [batch predictions]    [UI predictions]
             |
             v
     predicted_results.csv
     expected_features.txt
```

---

## What Is Missing (For Guthrie / Full Cure Rate Model)

The SEER dataset used here has only 9 columns. A full cure rate and longevity model requires:

| Feature | Why Needed | Available in Guthrie? |
|---|---|---|
| Survival months | Time-to-event for longevity modeling | TBD |
| Vital status (alive/dead) | Survival endpoint | TBD |
| T stage (tumor depth) | Strongest staging predictor | TBD |
| N stage (lymph node) | Staging predictor | TBD |
| CEA level | Tumor marker correlated with metastasis | TBD |
| Tumor size (cm) | Larger = higher risk | TBD |
| Lymphovascular invasion | Presence indicates higher risk | TBD |
| MSI status | Genetic marker, affects prognosis | TBD |
| Treatment history | Affects survival outcomes | TBD |

When Guthrie data arrives, the plan is:
1. Run the current SEER-trained model on Guthrie patients to measure accuracy
2. Use Guthrie's richer features to retrain or extend the model
3. Shift the target from binary metastasis to **survival time** (cure rate / longevity) using survival analysis methods (Cox regression or mixture cure models)

---

## How to Run (Order of Operations)

```bash
# Step 1: (One time) Convert raw SEER export to CSV
python load.py

# Step 2: Train the model (run after any data or feature changes)
python seer_model.py

# Step 3a: Predict for new patients (batch)
# Populate new_patients.csv first, then:
python seer_predict.py

# Step 3b: Run the doctor-facing web app
streamlit run app.py
```
