"""Render compact manuscript figures from the saved experiment CSV files."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "paper"
DATASETS = [
    "sonar_scale", "heart_scale", "australian_scale",
    "diabetes_scale", "fourclass_scale", "german.numer_scale",
]
NAMES = ["Sonar", "Heart", "Australian", "Diabetes", "Fourclass", "German.numer"]
C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
RATIOS = [0.1, 1 / 3, 1.0, 3.0, 10.0]


def get_case(frame, dataset, C, ratio):
    part = frame.loc[
        (frame.dataset == dataset)
        & np.isclose(frame.C, C)
        & np.isclose(frame.gamma_ratio, ratio)
    ]
    if len(part) != 1:
        raise ValueError(f"Expected one row for {dataset}, C={C}, ratio={ratio}")
    return part.iloc[0]


def plot_rbf_intervals(frame, dataset):
    points = []
    for c_idx, C in enumerate(C_VALUES):
        for g_idx, ratio in enumerate(RATIOS):
            row = get_case(frame, dataset, C, ratio)
            points.append((c_idx * 5.7 + g_idx,
                           row.one_minus_chi_exp, row.one_minus_chi_scr))

    positive = [value for _, exp, scr in points for value in (exp, scr)
                if np.isfinite(value) and value > 0]
    floor = 10 ** (np.floor(np.log10(min(positive))) - 0.45)

    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.75, bottom=0.22)
    ax.set_yscale("log")
    ax.set_ylim(floor, 2.0)
    ax.set_xlim(-0.5, 27.3)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    for x, exp, scr in points:
        if np.isfinite(exp) and exp > 0 and np.isfinite(scr) and scr > 0:
            ax.plot([x, x], [scr, exp], color="0.25", linewidth=0.8, zorder=2)
        if np.isfinite(exp) and exp > 0:
            ax.plot(x, exp, "o", color="black", markersize=3.0, zorder=4)
        if np.isfinite(scr) and scr > 0:
            ax.plot(x, scr, "o", markerfacecolor="white", markeredgecolor="black",
                    markersize=5.3, markeredgewidth=0.9, zorder=3)
        else:
            ax.plot(x, floor, "v", color="0.4", markersize=4.5, zorder=3)

    for group in range(1, 5):
        ax.axvline(group * 5.7 - 0.85, color="0.75", linewidth=0.6)
    xs = [c_idx * 5.7 + g_idx for c_idx in range(5) for g_idx in range(5)]
    ax.set_xticks(xs)
    ax.set_xticklabels([".1", "1/3", "1", "3", "10"] * 5, fontsize=6.7)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel(r"$1-\chi$", fontsize=8)
    ax.set_xlabel(r"$\gamma/\gamma_0$", fontsize=8, labelpad=2)
    for c_idx, C in enumerate(C_VALUES):
        ax.text(c_idx * 5.7 + 2, 1.06, f"$C={C:g}$",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    handles = [
        Line2D([], [], marker="o", color="black", linestyle="none", markersize=4,
               label=r"$1-\chi_{\rm obs}$"),
        Line2D([], [], marker="o", color="black", markerfacecolor="white",
               linestyle="none", markersize=5, label=r"$1-\chi_{\rm scr}$"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.26),
              ncol=2, frameon=False, fontsize=8)
    fig.savefig(OUTPUT / f"{dataset}_rbf.png", dpi=300)
    plt.close(fig)


def plot_heatmap(frame, baseline, group):
    column = "one_minus_chi_" + baseline
    all_matrices = []
    for dataset in DATASETS:
        matrix = np.full((5, 5), np.nan)
        for i, ratio in enumerate(RATIOS):
            for j, C in enumerate(C_VALUES):
                row = get_case(frame, dataset, C, ratio)
                scr, other = row.one_minus_chi_scr, row[column]
                if np.isfinite(scr) and np.isfinite(other) and scr > 0 and other > 0:
                    matrix[i, j] = np.log10(scr / other)
        all_matrices.append(matrix)

    finite = np.concatenate([m.ravel() for m in all_matrices])
    vmax = max(1, np.nanmax(finite))
    start = 0 if group == "main" else 3
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.30))
    fig.subplots_adjust(left=0.075, right=0.88, bottom=0.22, top=0.84, wspace=0.13)
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#eeeeee")
    for j, ax in enumerate(axes):
        matrix = all_matrices[start + j]
        image = ax.imshow(np.ma.masked_invalid(matrix), origin="lower", aspect="equal",
                          cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(NAMES[start + j], fontsize=8, pad=3)
        ax.set_xticks(range(5))
        ax.set_xticklabels([".01", ".1", "1", "10", "100"], fontsize=6.6)
        ax.set_yticks(range(5))
        ax.set_yticklabels([".1", "1/3", "1", "3", "10"], fontsize=6.6)
        if j != 0:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i, k in np.argwhere(~np.isfinite(matrix)):
            ax.text(k, i, r"$\times$", color="0.4", ha="center", va="center", fontsize=9)
    fig.text(0.48, 0.04, r"$C$", ha="center", fontsize=8)
    fig.text(0.055, 0.53, r"$\gamma/\gamma_0$", rotation=90, va="center", fontsize=8)
    bar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.03)
    bar.ax.tick_params(labelsize=7)
    bar.set_label(r"$\log_{10}$ ratio", fontsize=8)
    fig.savefig(OUTPUT / f"heatmap_{baseline}_{group}.png", dpi=300)
    plt.close(fig)


def plot_permutations(frame):
    cases = [
        ("rbf", 100.0, 10.0, "Large box"),
        ("rbf", 1.0, 3.0, "Boundary RBF"),
        ("linear", 1.0, None, "Singular linear"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.79, wspace=0.23)
    for ax, (kernel, C, ratio, title) in zip(axes, cases):
        part = frame.loc[(frame.kernel == kernel) & np.isclose(frame.C, C)]
        if ratio is not None:
            part = part.loc[np.isclose(part.gamma_ratio, ratio)]
        if len(part) != 100:
            raise ValueError(f"Expected 100 permutations for {title}")
        values = part[["one_minus_chi_exp", "one_minus_chi_scr"]].to_numpy()
        for exp, scr in values:
            if exp > 0 and scr > 0:
                ax.plot([0, 1], [exp, scr], color="0.78", linewidth=0.45, zorder=1)
        for x in (0, 1):
            col = values[:, x]
            q25, median, q75 = np.percentile(col, [25, 50, 75])
            ax.scatter(np.full(len(col), x), col, s=5, facecolor="0.65" if x == 0 else "white",
                       edgecolor="0.6", linewidth=0.5, alpha=0.7, zorder=2)
            ax.hlines([q25, q75], x - 0.06, x + 0.06,
                      color="black", linewidth=1.6, zorder=3)
            ax.scatter(x, median, s=32, facecolor="black" if x == 0 else "white",
                       edgecolor="black", linewidth=1.1, zorder=4)
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
        if ax is axes[0]:
            ax.set_yticks([0.1, 0.2])
            ax.set_yticklabels(["0.1", "0.2"])
        ax.set_xlim(-0.28, 1.28)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["obs", "scr"], fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel(r"$1-\chi$", fontsize=8)
    fig.text(0.5, 0.92, "100 coordinate permutations of Sonar", ha="center", fontsize=8)
    fig.savefig(OUTPUT / "permutations.png", dpi=300)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(RESULTS / "all_results.csv")
    rbf = data.loc[data.kernel == "rbf"]
    for dataset in DATASETS:
        plot_rbf_intervals(rbf, dataset)
    for baseline in ("wl", "gs"):
        for group in ("main", "appendix"):
            plot_heatmap(rbf, baseline, group)
    plot_permutations(pd.read_csv(RESULTS / "permutations" / "permutation_results.csv"))


if __name__ == "__main__":
    main()
