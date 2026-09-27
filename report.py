import numpy as np
import pandas as pd
import osqp

from scipy import sparse
from pathlib import Path
from scipy.optimize import minimize
from tqdm.auto import tqdm
from itertools import product

from bounds import compute_all_estimates, generate_M_scr, strengthen_M_scr
from empirical import experiment_worst_case
from problem import load_dataset, gamma_med, diag_scaling, build_dual_matrix
from cyclic_cd import solve_cd, one_cycle_cd
from plotting import GAMMA_RATIOS, plot_dataset_intervals, plot_rbf_heatmaps


def calc_1_minus_chi(gamma, method):
    if gamma is None:
        return np.nan
    if method == 'WL':
        return 1.0 / (1.0 + gamma)
    else:
        return 1.0 / (1.0 + 2 * gamma)


def solve_reference(H, a, r, dataset, kernel, C, max_cycles=20000):
    u_star, f_star, ref_residual = solve_cd(H, a, r, max_cycles=max_cycles)

    is_linear = kernel == "linear"
    needs_precise_reference = is_linear and (
        (dataset == "heart_scale" and C >= 10)
        or (dataset == "german.numer_scale"
            and (np.isclose(C, 1.0) or np.isclose(C, 10.0)))
    )
    use_osqp = (
        is_linear and dataset == "german.numer_scale"
        and np.isclose(C, 100.0)
    )

    def objective_and_grad(u):
        g = H @ u - a
        return 0.5 * u @ H @ u - a @ u, g

    def gap_at(u):
        g = H @ u - a
        return float(np.sum(np.maximum(g * u, g * (u - r))))

    if ref_residual > 1e-8:
        if use_osqp:
            solver = osqp.OSQP()
            solver.setup(
                P=sparse.csc_matrix(0.5 * (H + H.T)), q=-a,
                A=sparse.eye(len(a), format="csc"),
                l=np.zeros_like(r), u=r,
                eps_abs=1e-9, eps_rel=1e-9, max_iter=100000,
                polishing=True, verbose=False,
            )

            solver.warm_start(x=u_star)
            res = solver.solve()

            allowed = res.info.status in (
                "solved", "solved inaccurate", "maximum iterations reached")
        else:
            res = minimize(
                objective_and_grad, u_star, method="L-BFGS-B", jac=True,
                bounds=list(zip(np.zeros_like(r), r)),
                options={"maxiter": 10000, "ftol": 1e-15,
                         "gtol": 1e-10, "maxls": 50},
            )
            allowed = True

        if allowed and res.x is not None and np.all(np.isfinite(res.x)):
            candidate = np.clip(res.x, 0.0, r)
            g = H @ candidate - a
            candidate_residual = np.linalg.norm(
                candidate - np.clip(candidate - g, 0.0, r)
            )

            if is_linear:
                gap_before, gap_after = gap_at(u_star), gap_at(candidate)
                accept = np.isfinite(gap_after) and 0 <= gap_after < gap_before

            else:
                accept = candidate_residual < ref_residual

            if accept and np.isfinite(candidate_residual):
                u_star = candidate.copy()
                f_star = objective_and_grad(u_star)[0]
                ref_residual = candidate_residual

    if is_linear and (ref_residual > 1e-8 or needs_precise_reference):
        u = u_star.copy()
        extra_cycles = (30000 if dataset ==
                        "german.numer_scale" and C >= 10 else 10000)

        for i in range(extra_cycles):
            u = one_cycle_cd(u, H, a, r)
            g = H @ u - a
            residual = np.linalg.norm(u - np.clip(u - g, 0.0, r))
            fw_gap = gap_at(u)

            # if i == 0 or (i + 1) % 1000 == 0:
            #     print(f"CCD extra {i + 1}: residual={residual:.3e}, "
            #           f"FW gap≈{fw_gap:.3e}")

            if not np.isfinite(residual):
                break

            if residual < ref_residual:
                u_star = u.copy()
                ref_residual = residual
                f_star = objective_and_grad(u_star)[0]

            if needs_precise_reference:
                if residual <= 1e-11 and 0 <= fw_gap <= 1e-10:
                    u_star = u.copy()
                    ref_residual = residual
                    f_star = objective_and_grad(u_star)[0]
                    break

            elif ref_residual <= 1e-8:
                break

    return u_star, f_star, ref_residual


def run_case(X, y, *, dataset, kernel, C, gamma=None, gamma_ratio=None,
             gamma_0=None, max_cycles=20000, verbose=False):
    if kernel not in ("linear", "rbf"):
        raise ValueError(f"Unsupported kernel: {kernel}")

    Q, _ = build_dual_matrix(X, y, kernel=kernel, gamma=gamma)
    H, a, T, tau, r = diag_scaling(Q, C)
    u_star, f_star, ref_residual = solve_reference(
        H, a, r, dataset=dataset, kernel=kernel, C=C,
        max_cycles=max_cycles,
    )

    is_linear = kernel == "linear"
    is_sonar_100 = (
        is_linear and dataset == "sonar_scale" and np.isclose(C, 100.0)
    )
    needs_rank_check = is_linear and (
        dataset == "australian_scale"
        or (dataset == "heart_scale" and C >= 10)
        or (dataset == "german.numer_scale" and C >= 1)
    )

    if is_sonar_100 or needs_rank_check:
        M_old = generate_M_scr(H, a, r)
        M_new, diagnostics = strengthen_M_scr(H, a, r, u_star, M_old)

        tqdm.write(f"Dataset: {dataset}, C = {C}")
        tqdm.write(f"Screening diagnostics: {diagnostics}")

        if needs_rank_check:
            free = (np.diag(M_new) - np.diag(H)) <= 0.0
            H_free = H[np.ix_(free, free)]

            if H_free.size and verbose:
                tqdm.write(f"H_free size: {len(H_free)}")
                tqdm.write(
                    f"Numerical rank of H_free: {np.linalg.matrix_rank(H_free)}")
                tqdm.write(
                    f"lambda_min(H_free): {np.linalg.eigvalsh(H_free)[0]:.3e}")

    result = compute_all_estimates(
        Q, C, u_ref=u_star, pd_tol=1e-12 if is_sonar_100 else 1e-10
    )
    R, _ = experiment_worst_case(
        f_star, u_star, H, T, a, r, ref_residual
    )

    chi_exp = float(R) if np.isfinite(R) else np.nan

    row = {
        "dataset": dataset, "kernel": kernel, "C": C,
        "gamma_ratio": gamma_ratio, "gamma": gamma, "gamma_0": gamma_0,
        "chi_exp": chi_exp,
        "chi_scr": np.nan if result["chi_SCR"] is None else float(result["chi_SCR"]),
        "chi_wl": np.nan if result["chi_WL"] is None else float(result["chi_WL"]),
        "chi_scr_old": result.get("chi_SCR_old"),
        "one_minus_chi_exp": 1.0 - chi_exp,
        "one_minus_chi_scr": calc_1_minus_chi(result["gamma_SCR"], "SCR"),
        "one_minus_chi_wl": calc_1_minus_chi(result["gamma_WL"], "WL"),
        "one_minus_chi_gs": calc_1_minus_chi(result["gamma_GS"], "GS"),
        "reference_residual": ref_residual,
        "lambda_min_H": np.linalg.eigvalsh(H)[0],
    }
    diagnostics = result.get("screening_diagnostics")
    if diagnostics is not None:
        row.update(diagnostics)

    return row


def launch_experiment_with_report(dataset_path, output_dir="interval_plots", show=True, verbose=False,
                                  C_values=(0.01, 0.1, 1.0, 10.0, 100.0), max_cycles=20000, gamma_ratios=GAMMA_RATIOS):
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataset_path.name
    X, y = load_dataset(dataset, cache_dir=dataset_path.parent)
    gamma_0 = float(gamma_med(X))

    results_data = []

    cases = list(product(C_values, gamma_ratios))

    for C, ratio in tqdm(cases, desc=dataset, unit="case"):
        gamma = gamma_0 * ratio

        if verbose:
            tqdm.write(f"{dataset}: C={C:g}, γ/γ₀={ratio:g}")

        row = run_case(
            X, y, dataset=dataset, kernel="rbf", C=C,
            gamma=gamma, gamma_ratio=ratio, gamma_0=gamma_0,
            max_cycles=max_cycles,
        )
        results_data.append(row)

        pd.DataFrame(results_data).to_csv(
            output_dir / f"{dataset}_intervals.csv",
            index=False,
        )

    df = pd.DataFrame(results_data)

    plot_dataset_intervals(
        df,
        dataset,
        output_dir=output_dir,
        show=show,
        C_values=C_values,
    )

    return df


def launch_all_datasets(experiment_dir='.', output_dir='interval_plots', show=True, datasets=None, C_values=(0.01, 0.1, 1.0, 10.0, 100.0), max_cycles=20000, gamma_ratios=GAMMA_RATIOS):
    root = Path(experiment_dir)
    names = list(datasets) if datasets is not None else [
        'sonar_scale', 'heart_scale', 'australian_scale',
        'diabetes_scale', 'fourclass_scale', 'german.numer_scale']
    frames = [launch_experiment_with_report(root / "data" / name, output_dir, show,
                                            C_values=C_values, max_cycles=max_cycles,
                                            gamma_ratios=gamma_ratios) for name in names]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(Path(output_dir)/'all_datasets_intervals.csv', index=False)
    return combined


def make_linear_table(details, output_dir="linear_tables", datasets=None,
                      C_values=(0.01, 0.1, 1.0, 10.0, 100.0), show=True):
    details = details.copy()

    if "kernel" in details.columns:
        details = details.loc[details["kernel"] == "linear"].copy()

    names = list(datasets) if datasets is not None else list(
        details["dataset"].unique())
    details = details.loc[
        details["dataset"].isin(names) & details["C"].isin(C_values)].copy()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    details.to_csv(out / "linear_details.csv", index=False)

    columns = pd.MultiIndex.from_product([['C'], list(C_values)])
    index = pd.MultiIndex.from_product(
        [names, ['1 − χ_exp', '1 − χ_SCR', '1 − χ_WL']], names=['Dataset', 'Bound'],)
    table = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)

    for row in details.to_dict("records"):
        for label, field in [
                ('1 − χ_exp', 'one_minus_chi_exp'), ('1 − χ_SCR', 'one_minus_chi_scr'),
                ('1 − χ_WL', 'one_minus_chi_wl'),]:

            table.loc[(row['dataset'], label), ('C', row['C'])] = row[field]
    table.to_csv(out/'linear_table.csv')

    styled = (table.style
              .format(lambda value: f'{value:.2e}', na_rep='—')
              .set_caption('Linear kernel: estimates of 1 − χ')
              .set_table_styles([
                  {'selector': 'th, td', 'props': [
                      ('padding', '9px 14px'), ('border', '1px solid #d1d5db')]},
                  {'selector': 'th', 'props': [
                      ('background-color', '#f3f4f6'), ('color', '#111827')]},
                  {'selector': 'td', 'props': [('text-align', 'right')]},
                  {'selector': 'caption', 'props': [
                      ('caption-side', 'top'), ('padding', '12px'), ('font-weight', 'bold')]},
              ]))
    html = styled.to_html() + ('<p>—: the bound is unavailable or failed numerical checks. '
                               'Diagnostics has been saved in the linear_details.csv '
                               'Zero values are displayed as 0.00e+00.</p>')

    if show:
        from IPython.display import display, HTML
        display(HTML(html))

    return table, details


def launch_linear_table(experiment_dir='.', output_dir='linear_tables', datasets=None,
                        C_values=(0.01, 0.1, 1.0, 10.0, 100.0), max_cycles=20000, show=True):
    if datasets:
        names = list(datasets)
    else:
        names = [
            'sonar_scale', 'heart_scale', 'australian_scale',
            'diabetes_scale', 'fourclass_scale', 'german.numer_scale']

    root = Path(experiment_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:

        X, y = load_dataset(name, cache_dir=root / "data")

        for C in tqdm(C_values, desc=name, unit="case"):

            row = run_case(X, y, dataset=name, kernel="linear",
                           C=C, max_cycles=max_cycles,
                           )
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / 'linear_details.csv', index=False)

    details = pd.DataFrame(rows)
    return make_linear_table(details, output_dir=out, datasets=names, C_values=C_values, show=show)


def run_experiments(experiment_dir=".", output_dir="results", datasets=None, kernels=("linear", "rbf"), C_values=(0.01, 0.1, 1.0, 10.0, 100.0), gamma_ratios=GAMMA_RATIOS, max_cycles=20000, show=True,):
    if isinstance(kernels, str):
        kernels = (kernels,)

    kernels = tuple(kernels)

    if not kernels or len(set(kernels)) != len(kernels):
        raise ValueError("Kernels must be nonempty and contain no duplicates")

    if any(kernel not in ("linear", "rbf") for kernel in kernels):
        raise ValueError("Supported kernels: linear, rbf")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = []

    for kernel in kernels:
        if kernel == "linear":
            _, frame = launch_linear_table(
                experiment_dir=experiment_dir, output_dir=out / "linear",
                datasets=datasets, C_values=C_values,
                max_cycles=max_cycles, show=show
            )
        else:
            frame = launch_all_datasets(
                experiment_dir=experiment_dir, output_dir=out / "rbf",
                datasets=datasets, C_values=C_values,
                max_cycles=max_cycles, gamma_ratios=gamma_ratios, show=show
            )
        frames.append(frame)

        pd.concat(frames, ignore_index=True).to_csv(
            out / "all_results.csv", index=False)

    combined = pd.concat(frames, ignore_index=True)

    if "rbf" in kernels:
        plot_rbf_heatmaps(combined, output_dir=out / "heatmaps", show=show)

    lam_mins = combined[
        ["dataset", "kernel", "C", "gamma_ratio", "gamma", "lambda_min_H"]
    ].copy()
    lam_mins.to_csv(out / "min_eigenvalues.csv", index=False)

    return combined
