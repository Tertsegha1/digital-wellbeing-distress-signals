"""
Regenerates the descriptive visualisations from the original manuscript on
cleaned, de-duplicated data:

  * word clouds for depression-associated and comparison tweets
  * PCA (truncated SVD) "emotional landscape" of TF-IDF vectors
  * LDA topic bubble chart (topic size vs share of depression-associated tweets)
  * 3D behaviour space (screen time, sleep, TikTok)

Words used by fewer than --min-users distinct accounts are excluded from the
word clouds and topic model, so that account-specific hashtags, names and
handles cannot identify individuals or dominate the pictures.

Usage (from revision/):
    python analysis/emotional_visuals.py
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
from matplotlib import colormaps
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).parent))
from reanalysis import SEED, clean_tweet  # noqa: E402

LOWER, HIGHER = "#4C72B0", "#DD8452"
# Tokenisation artefacts (contraction fragments, anonymisation placeholder, HTML remnants)
ARTEFACTS = {"don", "didn", "doesn", "isn", "wasn", "won", "ain", "user", "amp"}


def shared_vocabulary(texts, users, min_users):
    """Tokens (len >= 3, not stop words) used by at least `min_users` distinct users."""
    seen = {}
    for t, u in zip(texts, users):
        for w in set(t.split()):
            if len(w) >= 3 and w not in ENGLISH_STOP_WORDS and w not in ARTEFACTS:
                seen.setdefault(w, set()).add(u)
    return {w for w, us in seen.items() if len(us) >= min_users}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="data/twitter/Mental-Health-Twitter.csv")
    ap.add_argument("--behaviour", default="data/behaviour/digital_habits_vs_mental_health.csv")
    ap.add_argument("--min-users", type=int, default=5)
    ap.add_argument("--topics", type=int, default=6)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--stats", default="analysis/output/emotional_visuals.json")
    a = ap.parse_args()
    out = Path(a.out)
    stats = {}

    df = pd.read_csv(a.text)
    df["clean"] = df["post_text"].map(clean_tweet)
    df = df[df["clean"].str.len() > 0].drop_duplicates(subset="clean").reset_index(drop=True)
    y = df["label"].astype(int).to_numpy()
    vocab = shared_vocabulary(df["clean"], df["user_id"], a.min_users)
    stats["shared_vocabulary_size"] = len(vocab)

    # ---------------- word clouds (shared vocabulary only) ----------------
    cv = CountVectorizer(vocabulary=sorted(vocab), token_pattern=r"(?u)\b\w\w\w+\b")
    counts = cv.fit_transform(df["clean"])
    terms = np.array(cv.get_feature_names_out())
    top = {}
    for lab, name, cmap in [(1, "fig_wordcloud_depression.png", "Oranges"),
                            (0, "fig_wordcloud_comparison.png", "Blues")]:
        freq = np.asarray(counts[y == lab].sum(axis=0)).ravel()
        d = {t: int(f) for t, f in zip(terms, freq) if f > 0}
        top[str(lab)] = [t for t, _ in sorted(d.items(), key=lambda kv: -kv[1])[:25]]
        cm = colormaps[cmap]
        palette = [matplotlib.colors.to_hex(cm(v)) for v in np.linspace(0.45, 1.0, 8)]
        pick = np.random.default_rng(SEED)

        def color_func(*_args, **_kw):
            return palette[pick.integers(len(palette))]

        wc = WordCloud(width=1600, height=900, background_color="white", color_func=color_func,
                       max_words=120, random_state=SEED, prefer_horizontal=0.95).generate_from_frequencies(d)
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.savefig(out / name, dpi=400, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    stats["top_terms_by_frequency"] = top

    # ---------------- PCA emotional landscape ----------------
    tfidf = TfidfVectorizer(vocabulary=sorted(vocab), token_pattern=r"(?u)\b\w\w\w+\b", sublinear_tf=True)
    X = tfidf.fit_transform(df["clean"])
    pca = PCA(n_components=2, svd_solver="arpack", random_state=SEED)
    Z = pca.fit_transform(X)
    stats["pca_explained_variance_pct"] = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(5000, len(df)), replace=False)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for lab, name, col in [(0, "Comparison", LOWER), (1, "Depression-associated", HIGHER)]:
        s = idx[y[idx] == lab]
        ax.scatter(Z[s, 0], Z[s, 1], s=4, alpha=0.45, color=col, label=name, linewidths=0)
    ax.set_xlabel(f"Component 1 ({stats['pca_explained_variance_pct'][0]:.1f}% variance)")
    ax.set_ylabel(f"Component 2 ({stats['pca_explained_variance_pct'][1]:.1f}%)")
    ax.legend(frameon=False, fontsize=7, markerscale=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_pca_landscape.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # how separable are the classes in 2D? centroid distance relative to spread
    c1, c0 = Z[y == 1].mean(axis=0), Z[y == 0].mean(axis=0)
    pooled_sd = np.sqrt((Z[y == 1].var(axis=0) + Z[y == 0].var(axis=0)) / 2)
    stats["pca_standardised_centroid_difference"] = [round(float(v), 3) for v in (c1 - c0) / pooled_sd]
    terms_pca = np.array(tfidf.get_feature_names_out())
    stats["pca_loadings"] = {
        f"PC{k + 1}": {"negative": terms_pca[np.argsort(pca.components_[k])[:6]].tolist(),
                       "positive": terms_pca[np.argsort(pca.components_[k])[-6:]].tolist()} for k in range(2)}
    regions = {"core (|PC1|,|PC2| < 0.05)": (np.abs(Z[:, 0]) < 0.05) & (np.abs(Z[:, 1]) < 0.05),
               "mental-health arm (PC2 > 0.1, PC1 < -0.02)": (Z[:, 1] > 0.1) & (Z[:, 0] < -0.02),
               "affect arm (PC1 > 0.15)": Z[:, 0] > 0.15}
    stats["pca_regions"] = {name: {"n_tweets": int(m.sum()), "pct_of_tweets": round(float(m.mean()) * 100, 1),
                                   "share_depression": round(float(y[m].mean()), 3),
                                   "n_users": int(df.loc[m, "user_id"].nunique())}
                            for name, m in regions.items()}

    # ---------------- LDA topic bubble chart ----------------
    lda_cv = CountVectorizer(vocabulary=sorted(vocab), token_pattern=r"(?u)\b\w\w\w+\b")
    C = lda_cv.fit_transform(df["clean"])
    lda = LatentDirichletAllocation(n_components=a.topics, random_state=SEED, learning_method="batch",
                                    max_iter=30)
    theta = lda.fit_transform(C)
    words = np.array(lda_cv.get_feature_names_out())
    assign = theta.argmax(axis=1)
    has_words = np.asarray(C.sum(axis=1)).ravel() > 0
    rows = []
    for k in range(a.topics):
        m = (assign == k) & has_words
        top_w = words[np.argsort(lda.components_[k])[::-1][:5]].tolist()
        rows.append({"topic": k + 1, "top_words": top_w, "n_tweets": int(m.sum()),
                     "share_depression": round(float(y[m].mean()), 3) if m.any() else None,
                     "n_users": int(df.loc[m, "user_id"].nunique())})
    stats["topics"] = rows
    stats["overall_share_depression"] = round(float(y.mean()), 3)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    t = pd.DataFrame(rows).sort_values("share_depression").reset_index(drop=True)
    sizes = t["n_tweets"] / t["n_tweets"].max() * 1500
    sc = ax.scatter(t.index, t["share_depression"], s=sizes, c=t["share_depression"], cmap="coolwarm",
                    vmin=0.2, vmax=0.8, alpha=0.75, edgecolors="grey")
    ax.axhline(y.mean(), color="grey", ls="--", lw=1)
    ax.text(-0.5, y.mean() + 0.012, "corpus average", ha="left", fontsize=7, color="grey")
    ax.set_xticks(t.index, [f"Topic {k}\n" + "\n".join(r[:3]) for k, r in zip(t["topic"], t["top_words"])],
                  fontsize=7)
    for i, r in t.iterrows():
        ax.annotate(f"n={r['n_tweets']:,}", (i, r["share_depression"]), ha="center", va="center", fontsize=7)
    ax.set_xlim(-0.6, len(t) - 0.4)
    ax.set_ylim(0.2, 0.85)
    ax.set_ylabel("Share of depression-associated tweets")
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(sc, ax=ax, pad=0.01).set_label("Share depression-associated", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_topic_bubble.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 3D behaviour space ----------------
    b = pd.read_csv(a.behaviour)
    yb = (b["stress_level"] >= 6).astype(int).to_numpy()
    s = rng.choice(len(b), size=3000, replace=False)
    fig = plt.figure(figsize=(3.6, 3.3))
    ax = fig.add_subplot(projection="3d")
    for lab, name, col in [(1, "Higher stress", "#D2541B"), (0, "Lower stress", "#1F4E9C")]:
        q = s[yb[s] == lab]
        ax.scatter(b.loc[q, "screen_time_hours"], b.loc[q, "sleep_hours"], b.loc[q, "hours_on_TikTok"],
                   s=4, alpha=0.6, color=col, label=name, linewidths=0, depthshade=False)
    ax.set_xlabel("Screen time (h/day)", fontsize=7, labelpad=-2)
    ax.set_ylabel("Sleep (h/night)", fontsize=7, labelpad=-2)
    ax.set_zlabel("TikTok (h/day)", fontsize=7, labelpad=-4)
    ax.tick_params(labelsize=6, pad=-2)
    ax.view_init(elev=20, azim=-60)
    ax.legend(frameon=False, fontsize=7, markerscale=3, loc="upper left")
    fig.savefig(out / "fig_behaviour_3d.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    stats["behaviour_corr_screen_tiktok"] = round(float(b["screen_time_hours"].corr(b["hours_on_TikTok"])), 3)

    Path(a.stats).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
