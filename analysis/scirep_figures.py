"""
Composite figure for the Scientific Reports version (one image file per figure,
as the Springer Nature template requires):

  fig_language_composite.png  (a) word cloud, depression-associated tweets
                              (b) word cloud, comparison tweets
                              (c) PCA emotional landscape

Built from the panels written by emotional_visuals.py.

Usage (from the repository root):  python analysis/scirep_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

FIG = Path("figures")


def panel(ax, path, letter, title):
    ax.imshow(mpimg.imread(FIG / path))
    ax.set_axis_off()
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=10, fontweight="bold")


def main():
    fig = plt.figure(figsize=(7.2, 4.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.18, wspace=0.04)
    panel(fig.add_subplot(grid[0, 0]), "fig_wordcloud_depression.png", "a", "Depression-associated tweets")
    panel(fig.add_subplot(grid[1, 0]), "fig_wordcloud_comparison.png", "b", "Comparison tweets")
    panel(fig.add_subplot(grid[:, 1]), "fig_pca_landscape.png", "c", "PCA emotional landscape")
    fig.savefig(FIG / "fig_language_composite.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (FIG / "fig_language_composite.png").resolve())


if __name__ == "__main__":
    main()
