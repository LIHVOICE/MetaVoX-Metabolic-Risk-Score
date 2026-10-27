# MetaVoX - Vocal Biomarker for Metabolic Risk

A voice-based Mixture-of-Experts pipeline that estimates cumulative cardiometabolic risk (T2D, hypertension, obesity, dyslipidaemia) from speech, attributing each participant’s risk score to the specific condition(s) driving it.

## Repository structure

```

├── help/                           # utilities shared across notebooks
│   ├── ckm_utils.py                # burden_score + 4 condition flags
│   ├── stats_utils.py              # bootstrap, Youden threshold, calibration, nested-CV splits
│   └── system_defs.py              # Mental/Respiratory control-group definitions
├── 01_train.ipynb                  # trains the 4 experts + gate, nested CV → oof_predictions.csv 
├── 02_baselines.ipynb              # non-vocal and single-model benchmarks on identical folds
├── 03_validation.ipynb             # discrimination, calibration, attribution, gate diagnostics
├── 04_stratification.ipynb         # performance by age/lifestyle factors
├── 05_specificity.ipynb            # off-target specificity, CVD positive control
├── 06_misclassifications.ipynb     # per-expert and gate flagging error (false positives/negatives)
├── 07_clinical_utility.ipynb       # triage zones, PPV/NPV
└── requirements.txt
```

Each analysis notebook writes its CSVs/pickles to its own `Results_*/` directory.

All reported performance estimates are derived exclusively from out-of-fold predictions generated under nested cross-validation. No participant used for evaluation contributed to model fitting, hyperparameter selection, calibration, or threshold estimation for that prediction.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10.

## Run order

1. `01_train.ipynb` - must run first; completes a full nested CV and generates `Results_MoE/oof_predictions.csv`, `Results_MoE/moe_participant_results.pkl`, auxiliary thresholds and diagnostic outputs; all downstream analyses operate exclusively on these out-of-fold predictions.
2. `02_baselines.ipynb`, `03_validation.ipynb`, `04_stratification.ipynb`, `05_specificity.ipynb`, `06_misclassifications.ipynb`, `07_clinical_utility.ipynb` - each reads `Results_MoE/oof_predictions.csv` (some also read `moe_participant_results.pkl`); run in any order.

## Provenance of the input data

**This repository begins at the analytic dataset.** Two upstream processing stages sit before it:

1. **Clinical harmonisation** - merging and cleaning harmonised questionnaire extractions, questionnaire scoring, and construction of the disease, BMI, and lifestyle flags, with eligibility and plausibility exclusions applied.
2. **Audio-clinical linkage and quality control** - matching recordings to clinical records, voice-activity and signal-to-noise screening, and filtering of the audio embeddings.

These two stages operate on raw, identifiable participant data and institutional storage and are **not part of this release**. Every exclusion and its count are documented in the manuscript (Figure S1 and corresponding Methods section). These stages produce exactly the de-identified artifacts this repository takes as input: `analytic_population.pkl`. This file contains participant identifiers, demographic variables, voice embedding features, disease labels, questionnaire-derived covariates, and the burden_score used throughout the analysis.

## Data availability

Raw participant-level voice recordings and identifiable clinical data are not publicly available, per participant privacy and consent terms. Audio embeddings are extracted from recordings collected from [Colive Voice](https://www.colivevoice.org/en/) participants (NCT04848623).

## Reproducibility notes

- All stochastic steps (train/test splits, bootstrap resampling) use a fixed `SEED = 42` from `help/stats_utils.py`.
- Package versions are pinned in `requirements.txt` to those used for the reported analysis.

All numerical results reported in the manuscript and supplementary appendix are generated directly from the notebooks contained in this repository. Figures and manuscript tables are derived solely from the CSV and pickle outputs produced by these notebooks.All analyses reported in the manuscript and supplementary appendix are generated from the notebooks contained in this repository.
