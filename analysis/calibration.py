"""
Calibration of every primary model (TRIPOD+AI items 12e and 23a).

Each model is refitted exactly as in reanalysis.py / behaviour_productivity.py (same data preparation,
splits, seeds and selected hyperparameters). Before any calibration statistic is computed, the script
checks that the refitted model reproduces the published AUC to six decimal places, so the calibration
refers to the same predictions as the reported discrimination.

Reported per model: Brier score, scaled Brier score (1 - Brier/Brier of a prevalence-only model),
calibration-in-the-large (intercept with the logit of the prediction as offset) and calibration slope,
with bootstrap 95% CIs (cluster bootstrap by user for the user-grouped language model), plus a
decile calibration plot (Supplementary Fig. S4).

Usage (from the repository root):  python analysis/calibration.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from reanalysis import C_GRID, SEED, clean_tweet, make_text_pipeline  # noqa: E402
from behaviour_productivity import BINARY, CATEGORICAL, NUMERIC, preprocessor  # noqa: E402

OUT = Path("analysis/output")
N_BOOT = 1000


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calib_stats(y, p):
    lp = logit(p)
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    brier = np.mean((p - y) ** 2)
    prev = y.mean()
    scaled = 1 - brier / (prev * (1 - prev))
    return {"brier": brier, "scaled_brier": scaled, "intercept": citl, "slope": slope}


def with_ci(y, p, groups=None, seed=SEED):
    point = calib_stats(y, p)
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in point}
    if groups is not None:
        uniq = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in uniq}
    for _ in range(N_BOOT):
        idx = (np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))]) if groups is not None
               else rng.integers(0, len(y), len(y)))
        if y[idx].min() == y[idx].max():
            continue
        try:
            for k, v in calib_stats(y[idx], p[idx]).items():
                boots[k].append(v)
        except Exception:  # rare non-convergence in a resample
            continue
    return {k: {"estimate": round(float(v), 6),
                "ci_low": round(float(np.nanpercentile(boots[k], 2.5)), 6),
                "ci_high": round(float(np.nanpercentile(boots[k], 97.5)), 6)} for k, v in point.items()}


def check_auc(name, y, p, expected):
    auc = round(float(roc_auc_score(y, p)), 6)
    assert abs(auc - expected) < 1e-6, f"{name}: refitted AUC {auc} != published {expected}"
    print(f"  {name}: AUC {auc} matches published value")


def deciles(y, p, n=10):
    q = pd.qcut(p, n, labels=False, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    return d


def main():
    R = json.loads((OUT / "results.json").read_text())
    P = json.loads((OUT / "productivity_results.json").read_text())
    preds = {}

    # ---- language pathway ----
    tw = pd.read_csv("data/twitter/Mental-Health-Twitter.csv")
    tw["clean"] = tw["post_text"].map(clean_tweet)
    tw = tw[tw["clean"].str.len() > 0].drop_duplicates(subset="clean").reset_index(drop=True)
    y = (tw["label"].astype(int) == 1).astype(int).to_numpy()
    X = tw["clean"].to_numpy()
    groups = tw["user_id"].to_numpy()

    tr, te = train_test_split(np.arange(len(tw)), test_size=0.2, stratify=y, random_state=SEED)
    g = GridSearchCV(make_text_pipeline(), C_GRID, scoring="roc_auc",
                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1).fit(X[tr], y[tr])
    p = g.predict_proba(X[te])[:, 1]
    check_auc("language tweet-level", y[te], p, R["language"]["tweet_level_split"]["test"]["auc"]["estimate"])
    preds["Language, tweet-level split"] = (y[te], p, None)

    prob = np.zeros(len(tw))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for a, b in outer.split(X, y, groups):
        gs = GridSearchCV(make_text_pipeline(), C_GRID, scoring="roc_auc",
                          cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED), n_jobs=-1)
        gs.fit(X[a], y[a], groups=groups[a])
        prob[b] = gs.predict_proba(X[b])[:, 1]
    check_auc("language user-grouped", y, prob, R["language"]["user_grouped_cv"]["test"]["auc"]["estimate"])
    preds["Language, user-grouped CV"] = (y, prob, groups)

    # ---- Digital Habits ----
    bh = pd.read_csv("data/behaviour/digital_habits_vs_mental_health.csv")
    feats = ["screen_time_hours", "sleep_hours", "hours_on_TikTok", "social_media_platforms_used"]
    Xb = bh[feats].to_numpy(float)
    yb = (bh["stress_level"] >= 6).astype(int).to_numpy()
    tr, te = train_test_split(np.arange(len(bh)), test_size=0.2, stratify=yb, random_state=SEED)
    bp = R["behaviour"]["rf_best_params"]
    rf = Pipeline([("sc", StandardScaler()), ("rf", RandomForestClassifier(
        n_estimators=bp["rf__n_estimators"], max_depth=bp["rf__max_depth"],
        min_samples_split=bp["rf__min_samples_split"], random_state=SEED, n_jobs=-1))]).fit(Xb[tr], yb[tr])
    p = rf.predict_proba(Xb[te])[:, 1]
    check_auc("Digital Habits RF", yb[te], p, R["behaviour"]["rf_test"]["auc"]["estimate"])
    preds["Digital Habits, random forest"] = (yb[te], p, None)
    lr = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))]).fit(Xb[tr], yb[tr])
    p = lr.predict_proba(Xb[te])[:, 1]
    check_auc("Digital Habits LR", yb[te], p, R["behaviour"]["lr_test"]["auc"]["estimate"])
    preds["Digital Habits, logistic regression"] = (yb[te], p, None)

    # ---- Social Media vs Productivity ----
    pr = pd.read_csv("data/productivity/social_media_vs_productivity.csv").dropna(subset=["stress_level"]).reset_index(drop=True)
    for c in BINARY:
        pr[c] = pr[c].astype(int)
    yp = (pr["stress_level"] >= 6).astype(int).to_numpy()
    Xp = pr[NUMERIC + BINARY + CATEGORICAL]
    tr, te = train_test_split(np.arange(len(pr)), test_size=0.2, stratify=yp, random_state=SEED)
    pp = P["rf_best_params"]
    rfp = Pipeline([("prep", preprocessor(NUMERIC + BINARY, CATEGORICAL)), ("rf", RandomForestClassifier(
        n_estimators=pp["rf__n_estimators"], max_depth=pp["rf__max_depth"],
        min_samples_split=pp["rf__min_samples_split"], random_state=SEED, n_jobs=-1))]).fit(Xp.iloc[tr], yp[tr])
    p = rfp.predict_proba(Xp.iloc[te])[:, 1]
    check_auc("Social Media vs Productivity RF", yp[te], p, P["rf_test"]["auc"]["estimate"])
    preds["Social Media vs Productivity, random forest"] = (yp[te], p, None)
    lrp = Pipeline([("prep", preprocessor(NUMERIC + BINARY, CATEGORICAL)),
                    ("lr", LogisticRegression(max_iter=2000))]).fit(Xp.iloc[tr], yp[tr])
    p = lrp.predict_proba(Xp.iloc[te])[:, 1]
    check_auc("Social Media vs Productivity LR", yp[te], p, P["lr_test"]["auc"]["estimate"])
    preds["Social Media vs Productivity, logistic regression"] = (yp[te], p, None)

    # ---- calibration statistics and plot ----
    res = {}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
    for ax, (name, (yy, pp_, gg)) in zip(axes.ravel(), preds.items()):
        res[name] = with_ci(yy, pp_, groups=gg)
        res[name]["n"] = int(len(yy))
        d = deciles(yy, pp_)
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
        ax.plot(d["pred"], d["obs"], "o-", color="#C44E52" if "Language" in name else "#4C72B0", ms=4)
        s = res[name]
        ax.set_title(name.replace(", ", "\n"), fontsize=9, loc="left")
        ax.text(0.03, 0.97, f"slope {s['slope']['estimate']:.2f}\nintercept {s['intercept']['estimate']:.2f}\n"
                            f"Brier {s['brier']['estimate']:.3f}", va="top", fontsize=8, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability", fontsize=8)
        ax.set_ylabel("Observed proportion", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/fig_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    (OUT / "calibration.json").write_text(json.dumps(res, indent=2))
    for name, s in res.items():
        print(f"{name}: " + ", ".join(f"{k} {v['estimate']:.3f} ({v['ci_low']:.3f}-{v['ci_high']:.3f})"
                                      for k, v in s.items() if k != "n"))


if __name__ == "__main__":
    main()
