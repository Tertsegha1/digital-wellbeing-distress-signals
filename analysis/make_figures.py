"""
Builds the manuscript figures from analysis/output/results.json and the
behavioural CSV, without re-running the models.

Usage (from the revision/ folder):
    python analysis/make_figures.py --behaviour data/behaviour/digital_habits_vs_mental_health.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOWER, HIGHER = "#4C72B0", "#DD8452"


def kde_panel(ax, df, col, y, xlabel, letter):
    lo, hi = df[col].min(), df[col].max()
    for g, lab, colr in [(0, "Lower stress", LOWER), (1, "Higher stress", HIGHER)]:
        df.loc[y == g, col].plot.kde(ax=ax, label=lab, color=colr, lw=1.8)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(letter, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def cm_panel(ax, cm, labels, title):
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(m / m.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            pct = m[i, j] / m[i].sum()
            ax.text(j, i, f"{m[i, j]:,}\n({pct:.0%})", ha="center", va="center",
                    color="white" if pct > 0.6 else "black", fontsize=9)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels, rotation=90, va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="analysis/output/results.json")
    ap.add_argument("--behaviour", required=True)
    ap.add_argument("--threshold", type=float, default=6)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    r = json.loads(Path(a.results).read_text())
    L, B = r["language"], r["behaviour"]

    # Figure 2: behavioural distributions
    df = pd.read_csv(a.behaviour)
    y = (df["stress_level"] >= a.threshold).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    kde_panel(axes[0], df, "screen_time_hours", y, "Screen time (hours/day)", "a")
    kde_panel(axes[1], df, "sleep_hours", y, "Sleep duration (hours/night)", "b")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig2_behaviour_kde.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: effect of validation design (AUC and balanced accuracy with 95% CIs)
    rows = [
        ("Language\ntweet-level split", L["tweet_level_split"]["test"], "#9AA5B1"),
        ("Language\nuser-grouped CV", L["user_grouped_cv"]["test"], "#C44E52"),
        ("Behaviour\nrandom forest", B["rf_test"], HIGHER),
        ("Behaviour\nlogistic regression", B["lr_test"], "#E8B48A"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, metric, title in [(axes[0], "auc", "a  AUC"), (axes[1], "balanced_accuracy", "b  Balanced accuracy")]:
        for i, (lab, t, colr) in enumerate(rows):
            e, lo, hi = t[metric]["estimate"], t[metric]["ci_low"], t[metric]["ci_high"]
            ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt="o", color=colr, capsize=4, ms=7, lw=1.8)
            ax.text(hi + 0.012, i, f"{e:.2f}", va="center", fontsize=9)
        ax.axvline(0.5, color="grey", ls="--", lw=1)
        ax.set_xlim(0.4, 1.0)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks(range(len(rows)), [r_[0] for r_ in rows])
    axes[0].invert_yaxis()
    axes[0].text(0.505, -0.55, "chance", color="grey", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: confusion matrices (primary analyses)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    cm_panel(axes[0], L["user_grouped_cv"]["test"]["confusion_matrix"], ["Comparison", "Depression"],
             "a  Language (user-grouped CV)")
    cm_panel(axes[1], B["rf_test"]["confusion_matrix"], ["Lower", "Higher"], "b  Behaviour (random forest)")
    fig.tight_layout()
    fig.savefig(out / "fig4_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    column_figures(a.results, a.behaviour, a.out, a.threshold)
    print("figures written to", out.resolve())



def column_figures(results="analysis/output/results.json",
                   behaviour="data/behaviour/digital_habits_vs_mental_health.csv", out="figures", threshold=6):
    """Single-panel, column-width versions for the two-column IEEE layout."""
    out = Path(out)
    r = json.loads(Path(results).read_text())
    df = pd.read_csv(behaviour)
    y = (df["stress_level"] >= threshold).astype(int)
    for col, xlabel, name, loc in [("screen_time_hours", "Screen time (hours/day)", "fig_kde_screen_col.png", "upper right"),
                                   ("sleep_hours", "Sleep duration (hours/night)", "fig_kde_sleep_col.png", "upper left")]:
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        kde_panel(ax, df, col, y, xlabel, "")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
        ax.legend(frameon=False, fontsize=8, loc=loc)
        fig.tight_layout()
        fig.savefig(out / name, dpi=300, bbox_inches="tight")
        plt.close(fig)
    for cm, labels, title, name in [
        (r["behaviour"]["rf_test"]["confusion_matrix"], ["Lower", "Higher"], "Behavioural (random forest)",
         "fig_cm_behaviour_col.png"),
        (r["language"]["user_grouped_cv"]["test"]["confusion_matrix"], ["Comparison", "Depression"],
         "Emotional (user-grouped CV)", "fig_cm_language_col.png")]:
        fig, ax = plt.subplots(figsize=(3.4, 3.1))
        cm_panel(ax, cm, labels, title)
        fig.tight_layout()
        fig.savefig(out / name, dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
