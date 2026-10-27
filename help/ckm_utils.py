"""ckm_utils.py — single source of truth for burden_score and its 4 components."""
CONDITIONS = {
    "T2D":          "ckm_t2d",
    "Hypertension": "ckm_hta",
    "BMI>=30":      "ckm_bmi30",
    "Cholesterol":  "ckm_chol",
}
COND_NAMES = list(CONDITIONS.keys())
