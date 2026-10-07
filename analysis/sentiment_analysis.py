"""
Sentiment analysis of the emotional pathway with VADER (Hutto & Gilbert, 2014).

VADER is applied to lightly cleaned text (URLs, @mentions and RT removed) so
that capitalisation, punctuation and emoticons, which VADER uses, are kept.
The tweet set is the same de-duplicated set used in reanalysis.py.

Because tweets are clustered within 72 users, group differences are tested
  (1) at the user level (per-user means; Mann-Whitney U, Cohen's d), and
  (2) at the tweet level with a linear mixed model (random intercept per user).
A sentiment-only classifier is evaluated with the same nested user-grouped
cross-validation as the TF-IDF model.

Usage (from revision/):  python analysis/sentiment_analysis.py
"""

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, str(Path(__file__).parent))
from reanalysis import SEED, clean_tweet, evaluate  # noqa: E402

LOWER, HIGHER = "#4C72B0", "#DD8452"
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"\bRT\b:?", re.IGNORECASE)


def light_clean(t):
    t = URL_RE.sub(" ", str(t))
    t = MENTION_RE.sub(" ", t)
    t = RT_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def main(text="data/twitter/Mental-Health-Twitter.csv", out_fig="figures", out_json="analysis/output/sentiment.json"):
    df = pd.read_csv(text)
    df["clean"] = df["post_text"].map(clean_tweet)
    df = df[df["clean"].str.len() > 0].drop_duplicates(subset="clean").reset_index(drop=True)
    sia = SentimentIntensityAnalyzer()
    scores = pd.DataFrame([sia.polarity_scores(light_clean(t)) for t in df["post_text"]])
    df = pd.concat([df, scores], axis=1)
    df["polarity"] = np.select([df["compound"] >= 0.05, df["compound"] <= -0.05], ["positive", "negative"], "neutral")
    y = df["label"].astype(int).to_numpy()
    res = {"n_tweets": len(df), "n_users": int(df["user_id"].nunique())}

    # Tweet-level descriptives by group
    desc = {}
    for lab, name in [(1, "depression_associated"), (0, "comparison")]:
        g = df[df["label"] == lab]
        desc[name] = {
            "compound_mean": round(g["compound"].mean(), 3), "compound_sd": round(g["compound"].std(), 3),
            "neg_mean": round(g["neg"].mean(), 3), "pos_mean": round(g["pos"].mean(), 3),
            "pct_negative": round((g["polarity"] == "negative").mean() * 100, 1),
            "pct_neutral": round((g["polarity"] == "neutral").mean() * 100, 1),
            "pct_positive": round((g["polarity"] == "positive").mean() * 100, 1)}
    res["tweet_level"] = desc

    # User-level comparison
    u = df.groupby("user_id").agg(label=("label", "first"), compound=("compound", "mean"), neg=("neg", "mean"),
                                  pos=("pos", "mean"), pct_neg=("polarity", lambda s: (s == "negative").mean()),
                                  n=("compound", "size"))
    ul = {}
    for col in ["compound", "neg", "pos", "pct_neg"]:
        a, b = u.loc[u.label == 1, col], u.loc[u.label == 0, col]
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        ul[col] = {"depression_mean": round(a.mean(), 3), "depression_sd": round(a.std(), 3),
                   "comparison_mean": round(b.mean(), 3), "comparison_sd": round(b.std(), 3),
                   "cohens_d": round(float(cohens_d(a, b)), 2),
                   # rank-biserial r = 2U/(n1*n2) - 1; positive = higher in depression-associated users
                   "rank_biserial": round(float(2 * stat / (len(a) * len(b)) - 1), 2),
                   "mannwhitney_p": round(float(p), 4)}
    res["user_level"] = ul
    dep_u, comp_u = u[u.label == 1], u[u.label == 0]
    strong_neg = dep_u[dep_u["compound"] < -0.2]
    res["user_level_robustness"] = {
        "median_compound_depression": round(float(dep_u["compound"].median()), 3),
        "median_compound_comparison": round(float(comp_u["compound"].median()), 3),
        "n_depression_users_mean_compound_below_minus_0.2": int(len(strong_neg)),
        "of_which_with_1_to_4_tweets": int((strong_neg["n"] <= 4).sum()),
        "depression_mean_excluding_them": round(float(dep_u.loc[dep_u["compound"] >= -0.2, "compound"].mean()), 3)}

    # Mixed models (random intercept per user)
    mm = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in ["compound", "neg", "pos"]:
            fit = smf.mixedlm(f"{col} ~ label", df, groups=df["user_id"]).fit(reml=True)
            ci = fit.conf_int().loc["label"]
            var_u = float(fit.cov_re.iloc[0, 0])
            mm[col] = {"label_coef": round(float(fit.params["label"]), 4), "ci_low": round(float(ci[0]), 4),
                       "ci_high": round(float(ci[1]), 4), "p": round(float(fit.pvalues["label"]), 4),
                       "icc": round(var_u / (var_u + float(fit.scale)), 3)}
    res["mixed_model"] = mm

    # Sentiment-only classifier, nested user-grouped CV (same design as TF-IDF model)
    X = df[["compound", "neg", "neu", "pos"]].to_numpy()
    groups = df["user_id"].to_numpy()
    prob = np.zeros(len(df))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for a, b in outer.split(X, y, groups):
        gs = GridSearchCV(Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))]),
                          {"lr__C": [0.01, 0.1, 1, 10]}, scoring="roc_auc",
                          cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED), n_jobs=-1)
        gs.fit(X[a], y[a], groups=groups[a])
        prob[b] = gs.predict_proba(X[b])[:, 1]
    ev = evaluate(y, prob, groups=groups)
    res["sentiment_only_classifier_user_grouped"] = {k: ev[k] for k in ["auc", "balanced_accuracy", "sensitivity",
                                                                        "specificity"]}

    # Figure: (a) tweet-level polarity shares, (b) per-user mean compound
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6), gridspec_kw={"height_ratios": [1, 1.1]})
    ax = axes[0]
    cats = ["negative", "neutral", "positive"]
    w = 0.38
    for i, (lab, name, col) in enumerate([(0, "Comparison", LOWER), (1, "Depression-associated", HIGHER)]):
        vals = [desc["depression_associated" if lab else "comparison"][f"pct_{c}"] for c in cats]
        bars = ax.bar(np.arange(3) + (i - 0.5) * w, vals, w, color=col, label=name)
        ax.bar_label(bars, fmt="%.0f%%", fontsize=6, padding=1)
    ax.set_xticks(range(3), ["Negative", "Neutral", "Positive"], fontsize=8)
    ax.set_ylabel("Tweets (%)", fontsize=8)
    ax.set_ylim(0, max(max(d[f"pct_{c}"] for c in cats) for d in desc.values()) * 1.45)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    ax.set_title("a  Tweet polarity (VADER)", loc="left", fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    rng = np.random.default_rng(SEED)
    for i, (lab, col) in enumerate([(0, LOWER), (1, HIGHER)]):
        v = u.loc[u.label == lab, "compound"]
        ax.scatter(rng.normal(i, 0.06, len(v)), v, s=np.clip(u.loc[u.label == lab, "n"] / 8, 6, 60), color=col,
                   alpha=0.7, edgecolors="white", linewidths=0.4)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color="black", lw=1.5)
    ax.axhline(0, color="grey", ls=":", lw=0.8)
    ax.set_xticks([0, 1], [f"Comparison\n(n={int((u.label == 0).sum())} users)",
                           f"Depression-associated\n(n={int((u.label == 1).sum())} users)"], fontsize=7)
    ax.set_ylabel("Mean compound score", fontsize=8)
    ax.set_title("b  Per-user mean sentiment", loc="left", fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_fig) / "fig_sentiment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    Path(out_json).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
