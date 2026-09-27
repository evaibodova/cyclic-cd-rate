import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

from plotting import plot_permutation_comparison
from problem import load_dataset, gamma_med, diag_scaling, build_dual_matrix
from bounds import compute_all_estimates
from empirical import experiment_worst_case
from report import solve_reference, calc_1_minus_chi

GAMMA_RATIOS_PERM = [10.0, 3.0]


def launch_experiment_with_report(input_dir='.', output_dir="results/permutations", dataset='sonar_scale', show=True,
                                  C_values=(100.0, 1.0, 1), max_cycles=20000, gamma_ratios=GAMMA_RATIOS_PERM,
                                  perm_cnt=100):
    output_dir = Path(output_dir)

    dataset_path = Path(input_dir) / "data" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset_path.name
    X, y = load_dataset(dataset, cache_dir=dataset_path.parent)
    gamma_0 = float(gamma_med(X))

    rbf_params = [
        (gamma_ratios[i], C_values[i]) for i in range(len(gamma_ratios))
    ]
    cases = [
        ("rbf", C, ratio) for ratio, C in rbf_params
    ]
    linear_C = C_values[-1]
    cases.append(("linear", linear_C, None))
    rows = []

    for kernel, C, ratio in cases:
        gamma = gamma_0 * ratio if kernel == "rbf" else None
        Q, _ = build_dual_matrix(X, y, kernel=kernel, gamma=gamma)
        H, a, _, _, r = diag_scaling(Q, C)

        u_star, f_star, ref_residual = solve_reference(
            H, a, r, dataset=dataset, kernel=kernel, C=C,
            max_cycles=max_cycles,
        )

        for seed in tqdm(range(42, 42 + perm_cnt), desc=f"{dataset}, {kernel}, C={C:g}", unit="perm"):
            perm = np.random.default_rng(seed).permutation(len(a))

            H_perm = H[np.ix_(perm, perm)]
            a_perm = a[perm]
            r_perm = r[perm]
            u_star_perm = u_star[perm]
            T_perm = np.tril(H_perm, k=-1)
            Q_perm = Q[np.ix_(perm, perm)]

            bounds_perm = compute_all_estimates(
                Q_perm,
                C,
                u_ref=u_star_perm,
            )

            chi_exp_perm, _ = experiment_worst_case(
                f_star, u_star_perm,
                H_perm, T_perm, a_perm, r_perm,
                ref_residual)

            row = {
                "dataset": dataset,
                "kernel": kernel,
                "C": C,
                "gamma_ratio": ratio if ratio is not None else np.nan,
                "permutation_seed": seed,
                "one_minus_chi_exp": 1.0 - chi_exp_perm,
                "one_minus_chi_scr": calc_1_minus_chi(
                    bounds_perm["gamma_SCR"], "SCR"
                ),
                "one_minus_chi_wl": calc_1_minus_chi(
                    bounds_perm["gamma_WL"], "WL"
                ),
                "one_minus_chi_gs": calc_1_minus_chi(
                    bounds_perm["gamma_GS"], "GS"
                ),
                "permutation_seed": seed}
            rows.append(row)

        results = pd.DataFrame(rows)
        results.to_csv(
            output_dir / "permutation_results.csv",
            index=False,
        )

    plot_permutation_comparison(
        results, output_dir=output_dir, show=show
    )
    return results
