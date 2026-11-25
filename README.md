# MetaVoX: voice-based metabolic risk score

Code accompanying the manuscript "Voice reflects cumulative metabolic burden rather than single diseases supporting system-level screening for prevention" (Veloso, Despotovic, Elbéji, Fagherazzi), Colive Voice study (ClinicalTrials.gov NCT04848623).

MetaVoX is a gender-stratified mixture-of-experts (MoE) pipeline. It takes one 1024-dimensional TRILLsson voice embedding per participant and returns (i) a calibrated metabolic risk score (0-100) and (ii) a per-condition attribution for type 2 diabetes (T2D), hypertension, obesity (BMI >= 30) and dyslipidaemia. Four condition-specific experts (standard scaling, PCA retaining 95% variance, logistic regression with sigmoid calibration) feed a logistic-regression gate; expert outputs are z-scored against participants without metabolic conditions, and attribution is obtained with alpha-entmax (alpha = 1.5), post hoc.

## System requirements

- Python 3.10 with the package versions pinned in `requirements.txt` (numpy 1.23.5, pandas 2.3.3, scikit-learn 1.3.0, scipy 1.10.1, statsmodels 0.14.0, xgboost 3.2.0, matplotlib 3.7.2, seaborn 0.12.2, openpyxl 3.1.5; Jupyter for the notebooks).
- Tested with Python 3.10 and the package versions pinned in `requirements.txt`
- No non-standard hardware is required (no GPU). The full analysis uses all available CPU cores.

## Installation guide

```bash
git clone https://github.com/LIHVOICE/MetaVoX-Metabolic-Risk-Score.git
cd MetaVoX-Metabolic-Risk-Score
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Instructions for use

To run the pipeline on your own data, provide one pickled `pandas.DataFrame` named `analytic_population.pkl` in the repository root with:

| Column                                                | Content                                                                                        |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `participant_id`                                    | unique identifier                                                                              |
| `gender`                                            | `female` or `male` (models are fitted separately by gender)                                |
| `ckm_t2d`, `ckm_hta`, `ckm_bmi30`, `ckm_chol` | binary flags (0/1) for the four conditions                                                     |
| `burden_score`                                      | sum of the four flags (0-4)                                                                    |
| `y_any`                                             | 1 if `burden_score` >= 1, otherwise 0                                                        |
| embedding columns                                     | 1024 numeric TRILLsson features; their names are stored as a list in `df.attrs["feat_cols"]` |

Then run `01_train.ipynb`, followed by the analysis notebooks. Participants with `burden_score == 0` are the reference controls for training the experts and for z-scoring.

## Repository structure

```
help/
  ckm_utils.py         # four condition flags and burden score
  stats_utils.py       # constants (SEED = 42, 10 repeats x 5 folds, 1000 bootstrap resamples), bootstrap, Youden threshold, calibration, nested-CV splits
  system_defs.py       # mental-health and respiratory control-group definitions
01_train.ipynb               # experts + gate under repeated nested CV -> Results_MoE/oof_predictions.csv
02_baselines.ipynb           # non-vocal and single-model benchmarks on identical folds
03_validation.ipynb          # discrimination, calibration, attribution, gate diagnostics
04_stratification.ipynb      # performance by age, smoking and alcohol
05_specificity.ipynb         # mental and respiratory negative controls, cardiovascular positive control
06_misclassifications.ipynb  # false positives and false negatives of experts and gate
07_clinical_utility.ipynb    # illustrative triage zones, PPV and NPV
requirements.txt
```

Each analysis notebook writes its outputs to its own `Results_*/` folder. Run `01_train.ipynb` first; notebooks 02-07 read `Results_MoE/oof_predictions.csv` (some also read `Results_MoE/moe_participant_results.pkl`) and can be run in any order.

## Reproducing the manuscript results

All reported performance estimates are out-of-fold predictions from repeated nested cross-validation (10 repeats of 5 outer folds; inner 5-fold loop). Hyperparameter selection, probability calibration, z-score normalisation and Youden threshold estimation use only the outer training fold. Confidence intervals are percentile intervals from 1000 bootstrap resamples (seed 42), except Wilson intervals for proportions. Running notebooks 01-07 on the real analytic file generates the numbers in the manuscript and supplementary information. The real file cannot be released (see below).

## Provenance of the input data

This repository starts at the analytic dataset. Two upstream stages operate on raw, identifiable data held on institutional servers and are not part of this release:

1. Clinical harmonisation: merging and cleaning questionnaire extractions, questionnaire scoring, construction of disease, BMI and lifestyle flags, and eligibility and plausibility exclusions.
2. Audio-clinical linkage and quality control: matching recordings to clinical records, voice-activity and signal-to-noise screening, TRILLsson embedding extraction and filtering.

Every exclusion and its count is documented in the manuscript Methods and Supplementary Fig. 1.

## Data availability

Raw voice recordings, audio embeddings and participant-level clinical data are not publicly available because of participant privacy and consent restrictions. They were collected from [Colive Voice](https://www.colivevoice.org/en/) participants (NCT04848623). Summary data are available from the corresponding author on reasonable request, subject to approval by the Colive Voice study governance framework.

## License

This code is released under the MIT License (see `LICENSE`).
