"""Shared definitions of the Mental and Respiratory systems."""
import pandas as pd

RESPIRATORY_SUBGROUPS = {"COPD": "has_copd", "Asthma": "has_asthma"}
MENTAL_THRESHOLDS = {"phq9_score": 10, "gad_score": 10, "fss_score": 45}
MENTAL_MIN_CRITERIA = 2

def mental_flag(df, thresholds=MENTAL_THRESHOLDS, min_criteria=MENTAL_MIN_CRITERIA):
    """1 if >= min_criteria of (diagnosed depression + the scale cut-offs) are met AND >= min_criteria have data."""
    met = {"has_depression": df["has_depression"] == 1}
    avail = {"has_depression": df["has_depression"].notna()}
    for col, thr in thresholds.items():                  
        met[col], avail[col] = df[col] >= thr, df[col].notna()
    n_met, n_avail = pd.DataFrame(met).sum(axis=1), pd.DataFrame(avail).sum(axis=1)
    return ((n_met >= min_criteria) & (n_avail >= min_criteria)).astype(int)

def add_system_flags(df):
    """Return a copy with has_any_mental and has_any_respiratory (0/1)."""
    df = df.copy()
    df["has_any_mental"] = mental_flag(df)
    df["has_any_respiratory"] = (df[list(RESPIRATORY_SUBGROUPS.values())].sum(axis=1) > 0).astype(int)
    return df