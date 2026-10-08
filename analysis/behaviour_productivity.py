"""
Behavioural pathway, dataset B: "Social Media vs Productivity" (Mashayekhi, Kaggle; synthetic,
30,000 records) -- the dataset of the original project (suzanayounas/Mental-Health-Vs-Social-Media).

Same design as dataset A (reanalysis.py, run_behaviour):
  * outcome: self-reported stress >= 6 (primary); >= 5 and >= 7 (the original project's
    threshold) as sensitivity analyses; records with missing stress are excluded
  * predictors: digital and lifestyle behaviours only. Outcome-adjacent wellbeing measures
    (perceived/actual productivity, burnout days, job satisfaction) are excluded to avoid
    leakage, as mood was for dataset A. Demographics (age, gender, job type) are used for the
    subgroup fairness analysis, not as predictors.
  * missing predictor values: median (numeric) / most-frequent (categorical) imputation fitted
    inside the training pipeline
  * random forest tuned by 5-fold CV + logistic regression comparator, stratified 80/20 split,
    bootstrap 95% CIs, permutation importance, mutually adjusted odds ratios,
    PPV at population prevalence, subgroup performance by gender and job type
  * leakage-maximal check: a model using every variable, including the excluded ones

Usage (from the repository root):
    python analysis/behaviour_productivity.py --data data/productivity/social_media_vs_productivity.csv
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from reanalysis import SEED, evaluate, ppv_table  # noqa: E402

NUMERIC = ["daily_social_media_time", "sleep_hours", "screen_time_before_sleep", "number_of_notifications",
           "work_hours_per_day", "breaks_during_work", "coffee_consumption_per_day", "weekly_offline_hours"]
BINARY = ["uses_focus_apps", "has_digital_wellbeing_enabled"]
CATEGORICAL = ["social_platform_preference"]
EXCLUDED_OUTCOME_ADJACENT = ["perceived_productivity_score", "actual_productivity_score",
                             "days_feeling_burnout_per_month", "job_satisfaction_score"]
DEMOGRAPHICS = ["age", "gender", "job_type"]


def preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def smd(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/productivity/social_media_vs_productivity.csv")
    ap.add_argument("--threshold", type=float, default=6)
    ap.add_argument("--out", default="analysis/output")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    out, figdir = Path(a.out), Path(a.figures)

    raw = pd.read_csv(a.data)
    res = {"n_raw": len(raw), "n_missing_stress": int(raw["stress_level"].isna().sum()),
           "missing_pct_by_column": (raw.isna().mean() * 100).round(1)[lambda s: s > 0].to_dict()}
    df = raw.dropna(subset=["stress_level"]).reset_index(drop=True)
    for c in BINARY:
        df[c] = df[c].astype(int)
    res["n_analysed"] = len(df)
    y = (df["stress_level"] >= a.threshold).astype(int).to_numpy()
    res["threshold"] = a.threshold
    res["prevalence"] = round(float(y.mean()), 6)
    res["n_higher"] = int(y.sum())
    res["n_lower"] = int(len(y) - y.sum())
    res["stress_distribution"] = {str(int(k)): int(v) for k, v in df["stress_level"].value_counts().sort_index().items()}
    num_all = df.select_dtypes("number")
    res["max_abs_correlation_with_stress"] = round(float(num_all.corr()["stress_level"].drop("stress_level").abs().max()), 4)

    # descriptives by stress group
    rows = []
    for c in NUMERIC + BINARY + ["stress_level"]:
        hi, lo = df.loc[y == 1, c].dropna(), df.loc[y == 0, c].dropna()
        rows.append({"variable": c, "all_mean": df[c].mean(), "all_sd": df[c].std(), "higher_mean": hi.mean(),
                     "higher_sd": hi.std(), "lower_mean": lo.mean(), "lower_sd": lo.std(), "smd": smd(hi, lo)})
    pd.DataFrame(rows).round(6).to_csv(out / "productivity_descriptives.csv", index=False)

    feats_num, feats_cat = NUMERIC + BINARY, CATEGORICAL
    X = df[feats_num + feats_cat]
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rf = Pipeline([("prep", preprocessor(feats_num, feats_cat)),
                   ("rf", RandomForestClassifier(random_state=SEED, n_jobs=-1))])
    grid = GridSearchCV(rf, {"rf__n_estimators": [100, 300, 500], "rf__max_depth": [None, 10, 20],
                             "rf__min_samples_split": [2, 10]}, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X.iloc[idx_tr], y[idx_tr])
    res["rf_best_params"] = grid.best_params_
    res["rf_cv_auc_mean"] = round(float(grid.best_score_), 6)
    prob_rf = grid.predict_proba(X.iloc[idx_te])[:, 1]
    res["rf_test"] = evaluate(y[idx_te], prob_rf)
    majority = int(y[idx_tr].mean() >= 0.5)
    res["majority_baseline_accuracy"] = round(float((y[idx_te] == majority).mean()), 6)

    lr = Pipeline([("prep", preprocessor(feats_num, feats_cat)), ("lr", LogisticRegression(max_iter=2000))])
    lr.fit(X.iloc[idx_tr], y[idx_tr])
    res["lr_test"] = evaluate(y[idx_te], lr.predict_proba(X.iloc[idx_te])[:, 1])

    # leakage-maximal check: every column except the outcome
    all_num = NUMERIC + BINARY + EXCLUDED_OUTCOME_ADJACENT + ["age"]
    all_cat = CATEGORICAL + ["gender", "job_type"]
    Xall = df[all_num + all_cat]
    m_all = Pipeline([("prep", preprocessor(all_num, all_cat)),
                      ("rf", RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1))])
    m_all.fit(Xall.iloc[idx_tr], y[idx_tr])
    res["rf_test_all_variables"] = evaluate(y[idx_te], m_all.predict_proba(Xall.iloc[idx_te])[:, 1], n_boot=1000)

    # permutation importance (AUC drop), on the original predictor columns
    pi = permutation_importance(grid.best_estimator_, X.iloc[idx_te], y[idx_te], scoring="roc_auc",
                                n_repeats=20, random_state=SEED, n_jobs=-1)
    res["permutation_importance_auc_drop"] = {
        f: {"mean": round(float(m), 6), "sd": round(float(s), 6)}
        for f, m, s in sorted(zip(feats_num + feats_cat, pi.importances_mean, pi.importances_std), key=lambda t: -t[1])}

    # mutually adjusted odds ratios per unit (complete cases, numeric + binary behaviours)
    cc = df.dropna(subset=feats_num)
    ycc = (cc["stress_level"] >= a.threshold).astype(int)
    logit = sm.Logit(ycc, sm.add_constant(cc[feats_num].astype(float))).fit(disp=0)
    ors = pd.DataFrame({"OR": np.exp(logit.params), "CI_low": np.exp(logit.conf_int()[0]),
                        "CI_high": np.exp(logit.conf_int()[1]), "p": logit.pvalues}).drop("const")
    ors.round(6).to_csv(out / "productivity_odds_ratios.csv")
    res["odds_ratios_per_unit"] = ors.round(6).to_dict(orient="index")
    res["n_complete_cases_for_or"] = int(len(cc))

    # thresholds (>= 7 is the original project's definition)
    sens = {}
    for t in [5, 7]:
        yt = (df["stress_level"] >= t).astype(int).to_numpy()
        tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=yt, random_state=SEED)
        m = clone(grid.best_estimator_).fit(X.iloc[tr], yt[tr])
        e = evaluate(yt[te], m.predict_proba(X.iloc[te])[:, 1], n_boot=500)
        sens[str(t)] = {"prevalence": e["prevalence_test"], "auc": e["auc"],
                        "balanced_accuracy": e["balanced_accuracy"]["estimate"]}
    res["threshold_sensitivity"] = sens

    # subgroup (fairness) performance of the primary random forest on the test set
    test = df.iloc[idx_te]
    sub = {}
    for col in ["gender", "job_type"]:
        sub[col] = {}
        for g, ix in test.groupby(col).groups.items():
            pos = np.array([np.where(idx_te == i)[0][0] for i in ix])
            yt, pt = y[idx_te][pos], prob_rf[pos]
            if len(np.unique(yt)) < 2 or len(yt) < 100:
                continue
            e = evaluate(yt, pt, n_boot=500)
            sub[col][str(g)] = {"n": int(len(yt)), "auc": e["auc"], "balanced_accuracy": e["balanced_accuracy"]["estimate"],
                                "sensitivity": e["sensitivity"]["estimate"], "specificity": e["specificity"]["estimate"]}
    res["subgroups"] = sub

    t = res["rf_test"]
    pd.DataFrame(ppv_table(t["sensitivity"]["estimate"], t["specificity"]["estimate"], "behaviour_B")).to_csv(
        out / "ppv_population_productivity.csv", index=False)

    # KDE figure (dataset B): daily social media time and sleep by stress group
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, col, label, letter in [(axes[0], "daily_social_media_time", "Daily social media time (hours)", "a"),
                                   (axes[1], "sleep_hours", "Sleep duration (hours/night)", "b")]:
        lo_, hi_ = df[col].min(), df[col].quantile(0.995)
        for g, name, colr in [(0, "Lower stress", "#4C72B0"), (1, "Higher stress", "#DD8452")]:
            df.loc[y == g, col].dropna().plot.kde(ax=ax, label=name, color=colr, lw=1.8)
        ax.set_xlim(lo_, hi_)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(letter, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "fig_productivity_kde.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    (out / "productivity_results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ["n_analysed", "prevalence", "max_abs_correlation_with_stress", "rf_best_params",
                                          "rf_cv_auc_mean", "majority_baseline_accuracy"]}, indent=1, default=str))
    for name in ["rf_test", "lr_test", "rf_test_all_variables"]:
        e = res[name]
        print(name, "AUC", e["auc"], "| bal acc", e["balanced_accuracy"]["estimate"],
              "| sens", e["sensitivity"]["estimate"], "| spec", e["specificity"]["estimate"])
    print("thresholds", sens)
    print("permutation importance", res["permutation_importance_auc_drop"])
    print("subgroups", json.dumps(sub, indent=1))
    print("ORs", ors.round(3).to_dict(orient="index"))


if __name__ == "__main__":
    main()
