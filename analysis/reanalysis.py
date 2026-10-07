"""
Reanalysis script for the revised manuscript.

Regenerates every number marked \\confirm{...} in revision/main.tex using a
leakage-safe, reproducible pipeline:

  Language pathway   : text cleaning (RT, @handles, URLs), de-duplication,
                       user-grouped train/test split (when a user column
                       exists), TF-IDF + L2 logistic regression tuned by
                       5-fold CV on the training set only.
  Behavioural pathway: mood score excluded (outcome leakage), stratified
                       descriptives with standardised mean differences,
                       random forest tuned by 5-fold CV, permutation
                       importance, logistic regression odds ratios per hour,
                       threshold sensitivity analysis.
  Both               : bootstrap 95% CIs (incl. AUC), majority baseline,
                       PPV/NPV at population prevalences, figures.

Usage (from the revision/ folder):
    pip install -r analysis/requirements.txt
    python analysis/reanalysis.py --text data/Mental-Health-Twitter.csv \
        --behaviour data/digital_habits.csv --out analysis/output

Column names can be changed with the --text-col / --label-col / --user-col /
--stress-col options. Run with --help for details.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_BOOT = 5000
PREVALENCES = [0.05, 0.10, 0.20, 0.30]


# ----------------------------------------------------------------- metrics
def metrics_from_counts(tn, fp, fn, tp):
    n = tn + fp + fn + tp
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else np.nan
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom else np.nan
    return {
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": (rec + spec) / 2,
        "sensitivity": rec,
        "specificity": spec,
        "precision_ppv": prec,
        "npv": npv,
        "f1": f1,
        "mcc": mcc,
    }


def evaluate(y_true, y_prob, threshold=0.5, n_boot=N_BOOT, seed=SEED, groups=None):
    """Point estimates and percentile bootstrap 95% CIs (positive class = 1).

    If `groups` is given, a cluster bootstrap is used: whole groups (users) are
    resampled, because observations from the same user are not independent.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    point = metrics_from_counts(tn, fp, fn, tp)
    point["auc"] = roc_auc_score(y_true, y_prob)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in uniq}
    boot = {k: [] for k in point}
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        yt, yp, pr = y_true[idx], y_pred[idx], y_prob[idx]
        if yt.min() == yt.max():
            continue
        c = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        m = metrics_from_counts(*c)
        m["auc"] = roc_auc_score(yt, pr)
        for k, v in m.items():
            boot[k].append(v)
    out = {}
    for k, v in point.items():
        lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
        out[k] = {"estimate": round(float(v), 6), "ci_low": round(float(lo), 6), "ci_high": round(float(hi), 6)}
    out["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    out["n_test"] = int(n)
    out["prevalence_test"] = round(float(y_true.mean()), 4)
    out["bootstrap"] = "cluster (by user)" if groups is not None else "observation-level"
    return out


def ppv_table(sens, spec, name):
    rows = []
    for p in PREVALENCES:
        ppv = sens * p / (sens * p + (1 - spec) * (1 - p))
        npv = spec * (1 - p) / (spec * (1 - p) + (1 - sens) * p)
        rows.append({
            "pathway": name, "prevalence": p, "ppv": round(ppv, 3), "npv": round(npv, 3),
            "tp_per_1000": round(sens * p * 1000), "fp_per_1000": round((1 - spec) * (1 - p) * 1000),
        })
    return rows


def plot_cm(y_true, y_pred, labels, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=[0, 1]), display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------- language path
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"\brt\b")
ENTITY_RE = re.compile(r"&\w+;")
NONALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")


def clean_tweet(text):
    t = str(text).lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = ENTITY_RE.sub(" ", t)
    t = RT_RE.sub(" ", t)
    t = NONALPHA_RE.sub(" ", t)
    return SPACE_RE.sub(" ", t).strip()


def make_text_pipeline(C=1.0):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_features=50000,
                                  sublinear_tf=True)),
        ("lr", LogisticRegression(penalty="l2", C=C, solver="liblinear", max_iter=2000, random_state=SEED)),
    ])


C_GRID = {"lr__C": [0.01, 0.1, 1, 10]}


def run_language(args, out):
    df = pd.read_csv(args.text)
    report = {"n_raw": len(df)}
    df["clean"] = df[args.text_col].map(clean_tweet)
    df = df[df["clean"].str.len() > 0]
    report["n_after_empty_removed"] = len(df)
    df = df.drop_duplicates(subset="clean").reset_index(drop=True)
    report["n_after_dedup"] = len(df)
    y = (df[args.label_col].astype(int) == args.positive_label).astype(int).to_numpy()
    report["class_counts"] = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    X = df["clean"].to_numpy()
    base_acc = float(max(y.mean(), 1 - y.mean()))
    report["majority_baseline_accuracy"] = round(base_acc, 4)

    # (a) Original design: one tweet-level stratified 80/20 split. Users can appear
    #     on both sides, so this estimate is optimistic. Reported for comparison only.
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    g = GridSearchCV(make_text_pipeline(), C_GRID, scoring="roc_auc",
                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1).fit(X[tr], y[tr])
    prob_naive = g.predict_proba(X[te])[:, 1]
    report["tweet_level_split"] = {"best_params": g.best_params_, "test": evaluate(y[te], prob_naive)}
    plot_cm(y[te], (prob_naive >= 0.5).astype(int), ["Comparison", "Depression-assoc."],
            "Language: tweet-level split", out / "fig_cm_language_tweet_split.png")

    if not (args.user_col and args.user_col in df.columns):
        report["note"] = "no user column: only the tweet-level split could be run"
        return report

    # (b) Primary analysis: user-grouped nested cross-validation. Outer 5 folds hold
    #     out whole users; C is tuned by an inner grouped 5-fold CV on training users.
    groups = df[args.user_col].to_numpy()
    report["n_users"] = int(pd.Series(groups).nunique())
    report["users_per_class"] = {str(k): int(v) for k, v in
                                 pd.Series(groups).groupby(y).nunique().items()}
    report["tweets_per_user"] = pd.Series(groups).value_counts().describe().round(1).to_dict()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    prob = np.zeros(len(df))
    fold_info = []
    for k, (a, b) in enumerate(outer.split(X, y, groups)):
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        gs = GridSearchCV(make_text_pipeline(), C_GRID, scoring="roc_auc", cv=inner, n_jobs=-1)
        gs.fit(X[a], y[a], groups=groups[a])
        prob[b] = gs.predict_proba(X[b])[:, 1]
        fold_auc = roc_auc_score(y[b], prob[b]) if len(np.unique(y[b])) == 2 else float("nan")
        fold_info.append({"fold": k, "test_users": int(len(np.unique(groups[b]))), "test_tweets": int(len(b)),
                          "best_C": gs.best_params_["lr__C"], "auc": round(float(fold_auc), 6)})
    report["user_grouped_cv"] = {
        "design": "nested user-grouped 5-fold CV; pooled out-of-fold predictions; cluster bootstrap by user",
        "folds": fold_info,
        "test": evaluate(y, prob, groups=groups),
    }
    plot_cm(y, (prob >= 0.5).astype(int), ["Comparison", "Depression-assoc."],
            "Language: user-grouped CV", out / "fig_cm_language.png")

    # User-level classification: mean out-of-fold probability per user
    u = pd.DataFrame({"user": groups, "y": y, "p": prob}).groupby("user").agg(y=("y", "first"), p=("p", "mean"))
    tn, fp, fn, tp = confusion_matrix(u.y, (u.p >= 0.5).astype(int), labels=[0, 1]).ravel()
    ul = {k: round(float(v), 6) for k, v in metrics_from_counts(tn, fp, fn, tp).items()}
    ul["auc"] = round(float(roc_auc_score(u.y, u.p)), 6)
    ul["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    report["user_level"] = ul

    # Interpretation: coefficients from a model fitted to all data with the modal C
    best_c = pd.Series([f["best_C"] for f in fold_info]).mode().iloc[0]
    final = make_text_pipeline(C=best_c).fit(X, y)
    vocab = np.array(final.named_steps["tfidf"].get_feature_names_out())
    coef = final.named_steps["lr"].coef_[0]
    order = np.argsort(coef)
    pd.DataFrame({
        "depression_assoc_term": vocab[order[::-1][:25]], "coef": coef[order[::-1][:25]].round(3),
        "comparison_term": vocab[order[:25]], "coef_": coef[order[:25]].round(3),
    }).to_csv(out / "language_top_terms.csv", index=False)
    return report


# ------------------------------------------------------ behavioural path
def smd(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def run_behaviour(args, out):
    df = pd.read_csv(args.behaviour)
    report = {"n_raw": len(df)}
    feats = args.features
    missing = [c for c in feats + [args.stress_col] if c not in df.columns]
    if missing:
        raise SystemExit(f"Behaviour file is missing columns {missing}. Available: {list(df.columns)}")
    df = df.dropna(subset=feats + [args.stress_col])
    report["n_complete"] = len(df)
    report["features_used"] = feats
    report["excluded_features_note"] = "mood score excluded to avoid outcome leakage"
    y_all = (df[args.stress_col] >= args.threshold).astype(int)
    report["threshold"] = args.threshold
    report["prevalence_full"] = round(float(y_all.mean()), 4)

    # Descriptives stratified by stress group, with standardised mean differences
    rows = []
    for c in feats + [args.stress_col]:
        hi, lo = df.loc[y_all == 1, c], df.loc[y_all == 0, c]
        rows.append({"variable": c, "all_mean": df[c].mean(), "all_sd": df[c].std(),
                     "min": df[c].min(), "max": df[c].max(),
                     "higher_mean": hi.mean(), "higher_sd": hi.std(),
                     "lower_mean": lo.mean(), "lower_sd": lo.std(), "smd": smd(hi, lo)})
    pd.DataFrame(rows).round(6).to_csv(out / "behaviour_descriptives.csv", index=False)

    X = df[feats].to_numpy(float)
    y = y_all.to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rf = Pipeline([("sc", StandardScaler()), ("rf", RandomForestClassifier(random_state=SEED, n_jobs=-1))])
    grid = GridSearchCV(rf, {"rf__n_estimators": [100, 300, 500], "rf__max_depth": [None, 10, 20],
                             "rf__min_samples_split": [2, 10]}, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X[idx_tr], y[idx_tr])
    report["rf_best_params"] = grid.best_params_
    report["rf_cv_auc_mean"] = round(float(grid.best_score_), 4)
    prob = grid.predict_proba(X[idx_te])[:, 1]
    report["rf_test"] = evaluate(y[idx_te], prob)
    majority = int(y[idx_tr].mean() >= 0.5)
    report["majority_baseline_accuracy"] = round(float((y[idx_te] == majority).mean()), 4)

    report["n_duplicate_rows"] = int(df.duplicated().sum())

    # Leakage check: same tuned model with the mood score added as a predictor
    if args.mood_col in df.columns:
        Xm = df[feats + [args.mood_col]].to_numpy(float)
        m = clone(grid.best_estimator_).fit(Xm[idx_tr], y[idx_tr])
        report["rf_test_with_mood"] = evaluate(y[idx_te], m.predict_proba(Xm[idx_te])[:, 1], n_boot=500)

    # Logistic regression comparator (scikit-learn) on the same split
    lr = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))]).fit(X[idx_tr], y[idx_tr])
    report["lr_test"] = evaluate(y[idx_te], lr.predict_proba(X[idx_te])[:, 1])

    # Permutation importance (test set, AUC drop)
    pi = permutation_importance(grid.best_estimator_, X[idx_te], y[idx_te], scoring="roc_auc",
                                n_repeats=20, random_state=SEED, n_jobs=-1)
    report["permutation_importance_auc_drop"] = {
        f: {"mean": round(float(m), 6), "sd": round(float(s), 6)}
        for f, m, s in sorted(zip(feats, pi.importances_mean, pi.importances_std), key=lambda t: -t[1])}

    # Mutually adjusted odds ratios per unit (hour / platform) - full sample, for interpretation
    logit = sm.Logit(y, sm.add_constant(df[feats].astype(float))).fit(disp=0)
    ors = pd.DataFrame({"OR": np.exp(logit.params), "CI_low": np.exp(logit.conf_int()[0]),
                        "CI_high": np.exp(logit.conf_int()[1]), "p": logit.pvalues}).drop("const")
    ors.round(6).to_csv(out / "behaviour_odds_ratios.csv")
    report["odds_ratios_per_unit"] = ors.round(6).to_dict(orient="index")

    # Error analysis: where are mistakes made relative to the raw stress score?
    te_scores = df[args.stress_col].to_numpy()[idx_te]
    wrong = (prob >= 0.5).astype(int) != y[idx_te]
    report["error_rate_by_stress_score"] = {
        str(int(s)): round(float(wrong[te_scores == s].mean()), 6) for s in np.unique(te_scores)}

    # Threshold sensitivity analysis (RF with selected params, retrained per threshold)
    sens = {}
    for t in args.sensitivity_thresholds:
        yt = (df[args.stress_col] >= t).astype(int).to_numpy()
        if yt.min() == yt.max():
            continue
        a, b = train_test_split(np.arange(len(df)), test_size=0.2, stratify=yt, random_state=SEED)
        m = clone(grid.best_estimator_).fit(X[a], yt[a])
        e = evaluate(yt[b], m.predict_proba(X[b])[:, 1], n_boot=500)
        sens[str(t)] = {"prevalence": e["prevalence_test"], "auc": e["auc"]["estimate"],
                        "balanced_accuracy": e["balanced_accuracy"]["estimate"]}
    report["threshold_sensitivity"] = sens

    # Figures with readable labels
    for c, fname, xlabel in [(feats[0], "fig_kde_screen.png", "Screen time (hours/day)"),
                             (feats[1], "fig_kde_sleep.png", "Sleep duration (hours/night)")]:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for g, lab, col in [(0, "Lower stress", "#4C72B0"), (1, "Higher stress", "#DD8452")]:
            df.loc[y_all == g, c].plot.kde(ax=ax, label=lab, color=col)
        ax.set_xlim(df[c].min(), df[c].max())
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=300)
        plt.close(fig)
    plot_cm(y[idx_te], (prob >= 0.5).astype(int), ["Lower", "Higher"], "Behavioural pathway",
            out / "fig_cm_behaviour.png")
    return report


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="CSV of tweets")
    ap.add_argument("--text-col", default="post_text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive-label", type=int, default=1,
                    help="value of --label-col denoting the depression-associated class")
    ap.add_argument("--user-col", default="user_id", help="user id column for a grouped split ('' to disable)")
    ap.add_argument("--behaviour", help="CSV of digital habits")
    ap.add_argument("--features", nargs="+",
                    default=["screen_time_hours", "sleep_hours", "hours_on_TikTok", "social_media_platforms_used"],
                    help="behavioural predictors; first two must be screen time and sleep")
    ap.add_argument("--stress-col", default="stress_level")
    ap.add_argument("--mood-col", default="mood_score", help="excluded from predictors; used only in a leakage check")
    ap.add_argument("--threshold", type=float, default=6, help="stress >= threshold is 'higher stress'")
    ap.add_argument("--sensitivity-thresholds", type=float, nargs="+", default=[5, 6, 7])
    ap.add_argument("--out", default="analysis/output")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, ppv = {}, []
    if args.text:
        results["language"] = run_language(args, out)
        lang = results["language"]
        t = lang["user_grouped_cv"]["test"] if "user_grouped_cv" in lang else lang["tweet_level_split"]["test"]
        ppv += ppv_table(t["sensitivity"]["estimate"], t["specificity"]["estimate"], "language")
    if args.behaviour:
        results["behaviour"] = run_behaviour(args, out)
        t = results["behaviour"]["rf_test"]
        ppv += ppv_table(t["sensitivity"]["estimate"], t["specificity"]["estimate"], "behaviour")
    if not results:
        ap.error("give --text and/or --behaviour")
    pd.DataFrame(ppv).to_csv(out / "ppv_population.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    print(f"\nAll outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
