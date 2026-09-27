from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
GAMMA_RATIOS = [0.1, 1/3, 1.0, 3.0, 10.0]
GAMMA_LABELS = [r'$\gamma_0/10$', r'$\gamma_0/3$',
                r'$\gamma_0$', r'$3\gamma_0$', r'$10\gamma_0$']


def plot_dataset_intervals(df, dataset, output_dir='interval_plots', show=True, source_note='', C_values=C_VALUES):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for group, C in enumerate(C_values):
        for j, ratio in enumerate(GAMMA_RATIOS):
            found = df[np.isclose(df['C'], C) & np.isclose(
                df['gamma_ratio'], ratio)]

            if len(found) > 1:
                raise ValueError(
                    f'Duplicate case: {dataset}, C={C}, gamma_ratio={ratio}')

            row = found.iloc[0] if len(found) else None
            exp = float(row.chi_exp) if row is not None and pd.notna(
                row.chi_exp) else np.nan
            scr = float(row.chi_scr) if row is not None and pd.notna(
                row.chi_scr) else np.nan

            records.append((group * 5.3 + j, C, ratio, exp, scr))

    positive = [1 - v for row in records for v in row[3:]
                if np.isfinite(v) and 0 <= v < 1]
    exponent = min(-2, int(np.floor(np.log10(min(positive))))
                   if positive else -2)
    floor = 10.0 ** (exponent - 0.5)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11}):
        fig, ax = plt.subplots(figsize=(18, 6.6))

        ax.set_yscale('log')
        ax.set_ylim(floor, 2.5)
        ax.set_xlim(-0.8, 5.3 * (len(C_values) - 1) + 4.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', which='major', color='#d1d5db', linewidth=0.8)

        invalid = 0
        missing = 0
        zeros = 0

        for x, C, ratio, exp, scr in records:
            upper = 1-exp if np.isfinite(exp) and 0 <= exp <= 1 else np.nan
            lower = 1-scr if np.isfinite(scr) and 0 <= scr <= 1 else np.nan
            violation = np.isfinite(exp) and np.isfinite(
                scr) and exp > scr + 2e-7
            color = '#b91c1c' if violation else '#171717'
            invalid += int(violation)

            if np.isfinite(lower) and np.isfinite(upper) and lower > 0 and upper > 0:
                ax.plot([x, x], [lower, upper], color=color, lw=1.5, zorder=2)

            if np.isfinite(lower) and lower > 0:
                ax.plot(x, lower, 'o', ms=10.5, mfc='white',
                        mec=color, mew=1.6, zorder=3)
            elif not np.isfinite(lower):
                ax.plot(x, 0, marker='v', markersize=8, markerfacecolor='white', markeredgecolor='#6b7280',
                        markeredgewidth=1.5, linestyle='None', transform=ax.get_xaxis_transform(), clip_on=False, zorder=5,)

                if np.isfinite(upper) and upper > 0:
                    ax.plot([x, x], [ax.get_ylim()[0], upper], linestyle='--',
                            color='#6b7280', linewidth=1.2, zorder=1,)

            if np.isfinite(upper) and upper > 0:
                ax.plot(x, upper, 'o', ms=5.5, mfc=color, mec=color, zorder=4)

            tags = []

            for value, symbol in [(upper, '●'), (lower, '○')]:
                if not np.isfinite(value):
                    tags.append(symbol+' no')
                    missing += 1

                elif value == 0:
                    tags.append(symbol+' 0')
                    zeros += 1
            if tags:
                ax.text(x, -.135, '\n'.join(tags), transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=8, color='#6b7280', clip_on=False)

        labels = []

        for _, C, ratio, _, _ in records:
            row = df[np.isclose(df["C"], C) & np.isclose(
                df["gamma_ratio"], ratio)]

            j = np.flatnonzero(np.isclose(GAMMA_RATIOS, ratio))[0]
            gamma_label = GAMMA_LABELS[j]
            labels.append(gamma_label)

        ax.set_xticks(
            [row[0] for row in records],
            labels,
            fontsize=8,
        )

        for group, C in enumerate(C_values):
            center = group * 5.3 + 2
            ax.text(center, -0.255, f'C = {C:g}', transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=13, fontweight='bold')

            if group:
                ax.axvline(group * 5.3 - 0.65, color='#9ca3af', lw=1)

        ax.set_ylabel(r'$1-\chi$', fontsize=13)

        handles = [Line2D([], [], color='black', marker='o', linestyle='None', markersize=6,
                          label=r'$1-\chi_{\rm exp}$'),
                   Line2D([], [], color='black', marker='o', markerfacecolor='white', linestyle='None', markersize=9,
                          label=r'$1-\chi_{\rm SCR}$')]

        ax.legend(handles=handles, loc='lower center',
                  bbox_to_anchor=(.5, 1.025), ncol=2, frameon=False)

        if source_note:
            fig.text(0.5, 0.025, source_note, ha='center',
                     fontsize=9, color='#4b5563')

        fig.subplots_adjust(left=.065, right=.985, top=.77, bottom=0.30)

        safe_name = ''.join(
            c if c.isalnum() or c in '-_.' else '_' for c in dataset)

        png = output_dir/f'{safe_name}_intervals.png'
        fig.savefig(png, dpi=200)
        fig.savefig(png.with_suffix('.svg'))

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig, ax, png


def plot_permutation_comparison(df, output_dir="results/permutations", show=True):
    df = df.copy()
    df["gamma_ratio"] = pd.to_numeric(df["gamma_ratio"], errors="coerce")

    if df["dataset"].nunique() != 1:
        raise ValueError("The function works only for one dataset")

    cases = [
        ("rbf", 100.0, 10.0, r"$C=100,\ \gamma=10\gamma_0$"),
        ("rbf", 1.0, 3.0, r"$C=1,\ \gamma=3\gamma_0$"),
        ("linear", 1.0, None, r"$C=1$, linear SVM"),
    ]
    columns = ["one_minus_chi_exp", "one_minus_chi_scr"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)

    for ax, (kernel, C, ratio, title) in zip(axes, cases):
        mask = (df["kernel"] == kernel) & np.isclose(df["C"], C)

        if ratio is not None:
            mask &= np.isclose(df["gamma_ratio"], ratio)

        part = df.loc[mask].copy()

        if part["permutation_seed"].duplicated().any():
            raise ValueError(f"Permutations are duplicated for {title}")

        values = part[columns].to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0)

        for pair, ok in zip(values, valid):
            if ok.all():
                ax.plot(
                    [0, 1], pair,
                    color="0.75", alpha=0.6, linewidth=0.8,
                    zorder=1,
                )

        for x, filled in enumerate([True, False]):
            y = values[valid[:, x], x]

            ax.scatter(
                np.full(len(y), x), y,
                s=28,
                facecolors="0.65" if filled else "none",
                edgecolors="0.65",
                alpha=0.7,
                zorder=2,
            )

            if len(y):
                q25, median, q75 = np.percentile(y, [25, 50, 75])

                ax.vlines(x, q25, q75, color="black", linewidth=2)
                ax.hlines([q25, q75], x - 0.12, x + 0.12,
                          color="black", linewidth=2)
                ax.scatter(
                    x, median, s=90,
                    facecolors="black" if filled else "white",
                    edgecolors="black", linewidths=2,
                    zorder=4,
                )

        ax.set_yscale("log")

        positive_values = values[valid]
        if positive_values.size:
            log_min = np.log10(positive_values.min())
            log_max = np.log10(positive_values.max())

            padding = max(0.08, 0.15 * (log_max - log_min))

            ax.set_ylim(
                10 ** (log_min - padding),
                10 ** (log_max + padding),
            )

        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["exp", "scr"])
        ax.set_xlabel(title, labelpad=12)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel(r"$1-\chi$")

    handles = [
        Line2D([], [], marker="o", color="black", linestyle="none",
               label=r"$1-\chi_{\mathrm{exp}}$"),
        Line2D([], [], marker="o", color="black", markerfacecolor="white",
               linestyle="none", label=r"$1-\chi_{\mathrm{scr}}$"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.94), ncol=2)

    fig.tight_layout(rect=(0, 0.05, 1, 0.85))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "permutation_comparison.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(out / "permutation_comparison.svg",
                bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_rbf_heatmaps(results, output_dir="heatmaps", show=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rbf = results.loc[results["kernel"] == "rbf"].copy()

    datasets = [
        "sonar_scale",
        "heart_scale",
        "australian_scale",
        "diabetes_scale",
        "fourclass_scale",
        "german.numer_scale",
    ]
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    ratios = [0.1, 1 / 3, 1.0, 3.0, 10.0]
    gamma_labels = [
        r"$\gamma_0/10$",
        r"$\gamma_0/3$",
        r"$\gamma_0$",
        r"$3\gamma_0$",
        r"$10\gamma_0$",
    ]

    names = [
        ("one_minus_chi_wl", "Wang–Lin", "wl"),
        ("one_minus_chi_gs", "Gauss–Seidel", "gs"),
    ]

    for baseline, baseline_name, suffix in names:
        matrices = []

        for dataset in datasets:
            part = rbf.loc[rbf["dataset"] == dataset]
            matrix = np.full((5, 5), np.nan)

            for i, ratio in enumerate(ratios):
                for j, C in enumerate(C_values):
                    found = part.loc[
                        np.isclose(part["C"], C) & np.isclose(
                            part["gamma_ratio"], ratio)
                    ]

                    if len(found) > 1:
                        raise ValueError(
                            f"Duplicate: {dataset}, C={C}, ratio={ratio}")
                    if found.empty:
                        continue

                    row = found.iloc[0]
                    scr = pd.to_numeric(
                        row["one_minus_chi_scr"], errors="coerce")
                    other = pd.to_numeric(row[baseline], errors="coerce")

                    if np.isfinite(scr) and np.isfinite(other) and scr > 0 and other > 0:
                        matrix[i, j] = (np.log10(scr) - np.log10(other))

            matrices.append(matrix)

        all_values = np.concatenate([m.ravel() for m in matrices])
        finite = all_values[np.isfinite(all_values)]

        vmin = min(0.0, float(finite.min())) if finite.size else 0.0
        vmax = max(1.0, float(finite.max())) if finite.size else 1.0

        groups = [
            ("main", datasets[:3], matrices[:3]),
            ("appendix", datasets[3:], matrices[3:]),
        ]

        cmap = plt.get_cmap("Reds").copy()
        cmap.set_bad("#eeeeee")

        for group_name, group_datasets, group_matrices in groups:
            fig, axes = plt.subplots(
                1, 3,
                figsize=(13, 4.5),
                layout="constrained",
            )
            for index, (ax, dataset, matrix) in enumerate(
                zip(axes.flat, group_datasets, group_matrices)
            ):
                image = ax.imshow(
                    np.ma.masked_invalid(matrix),
                    origin="lower",
                    aspect="equal",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

                ax.set_title(dataset)
                ax.set_xticks(range(5))
                ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
                ax.set_yticks(range(5))
                ax.set_yticklabels(gamma_labels)

                ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
                ax.grid(which="minor", color="white", linewidth=1)
                ax.tick_params(which="minor", bottom=False, left=False)

                for i, j in np.argwhere(~np.isfinite(matrix)):
                    ax.text(
                        j, i, "×",
                        ha="center", va="center",
                        color="#666666", fontsize=15,
                    )

                ax.tick_params(
                    axis="x", which="major",
                    labelbottom=True,
                    bottom=True,
                )
                ax.tick_params(
                    axis="y", which="major",
                    labelleft=(index == 0),
                    left=(index == 0),
                )

            colorbar = fig.colorbar(
                image, ax=axes.ravel().tolist(), shrink=0.85)
            colorbar.set_label(
                r"$\log_{10}\!\left("
                r"\frac{1-\chi_{\mathrm{SCR}}}"
                rf"{{1-\chi_{{\mathrm{{{suffix.upper()}}}}}}}\right)$",
                fontsize=16
            )
            fig.supxlabel(r"$C$")
            fig.supylabel(r"$\gamma$")

            out = Path(output_dir)
            for extension in ("png", "svg"):
                fig.savefig(
                    out /
                    f"rbf_screening_vs_{suffix}_{group_name}.{extension}",
                    dpi=300,
                    bbox_inches="tight",
                )

            if show:
                plt.show()
            else:
                plt.close(fig)
