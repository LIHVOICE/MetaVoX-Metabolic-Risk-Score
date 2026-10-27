"""Shared constants and helpers"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.proportion import proportion_confint

# Repeated 5-fold nested CV (10 repeats) was used to reduce variance from a single random split, 
# 1000 bootstrap resamples were used for 95% percentile CIs,
SEED, N_REPEATS, N_OUTER, N_BOOT = 42, 10, 5, 1000

def outer_splits(y, n_repeats=N_REPEATS, n_splits=N_OUTER, seed=SEED):
    """Repeated stratified K-fold. Every model is scored on identical folds."""
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(np.zeros(len(y)), y):
            yield rep, tr, te

def youden_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]

def ece(y, p, n_bins=8):
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    return float(sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean())
                     for b in np.unique(bins)))

def brier_platt(y, p):
    """Brier after 2-parameter Platt recalibration"""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    q = LogisticRegression(C=1e6, max_iter=1000).fit(z, y).predict_proba(z)[:, 1]
    return brier_score_loss(y, q)

def score_metrics(y, s, thr=None):
    """AUC, sensitivity, specificity, Brier (raw / recalibrated), ECE.
    thr=None: Youden threshold is estimated on (y, s)"""
    y = np.asarray(y).astype(int)
    thr = youden_threshold(y, s) if thr is None else thr
    pred = s >= thr
    return {"AUC": roc_auc_score(y, s),
            "Sensitivity": pred[y == 1].mean(), "Specificity": (~pred[y == 0]).mean(),
            "Brier": brier_score_loss(y, s), "Brier (recalibrated)": brier_platt(y, s),
            "ECE": ece(y, s)}

def bootstrap(fn, *arrays, n_boot=N_BOOT, seed=SEED, return_reps=False):
    """Percentile 95% CI of fn(*arrays), resampling participants. fn returns {metric: value}."""
    arrays = [np.asarray(a) for a in arrays]
    n, rng = len(arrays[0]), np.random.default_rng(seed)
    estimate, reps = fn(*arrays), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            reps.append(fn(*[a[i] for a in arrays]))
        except ValueError:                       # resample had a single class
            continue
    reps = pd.DataFrame(reps)
    keys = list(estimate)
    table = pd.DataFrame({"metric": keys, "estimate": [estimate[k] for k in keys],
                          "ci_lo": reps[keys].quantile(0.025).to_numpy(),
                          "ci_hi": reps[keys].quantile(0.975).to_numpy(),
                          "n_valid_boot": len(reps)})
    return (table, reps) if return_reps else table

def calibration_table(y, p, n_bins=8):
    """Quantile-bin calibration curve with Wilson 95% CIs on the observed fraction."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    bins, rows = pd.qcut(p, n_bins, labels=False, duplicates="drop"), []
    for b in np.unique(bins):
        m = bins == b
        lo, hi = proportion_confint(y[m].sum(), m.sum(), method="wilson")
        rows.append({"bin": b, "n": int(m.sum()), "mean_predicted": p[m].mean(),
                     "observed": y[m].mean(), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)

def entmax15(z, n_iter=60):
    """alpha = 1.5 entmax by bisection (Peters et al. 2019); each row sums to 1."""
    zs = z * 0.5
    top = zs.max(axis=1, keepdims=True)
    lo, hi = top - 1, top - z.shape[1] ** -0.5
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        f = (np.clip(zs - mid, 0, None) ** 2).sum(axis=1, keepdims=True) - 1
        lo, hi = np.where(f < 0, lo, mid), np.where(f < 0, mid, hi)
    p = np.clip(zs - (lo + hi) / 2, 0, None) ** 2
    return p / p.sum(axis=1, keepdims=True)