import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar
import math
from problem import diag_scaling


def wang_lin(H, kappa):
    p = H.shape[0]
    rho = eigsh(H, k=1, which="LA", return_eigenvectors=False)[0]

    A_wl = (2 * (rho + 2 + rho * np.sqrt(p)) *
            (1 + kappa * (2 + rho * np.sqrt(p))))
    chi_wl = A_wl / (1 + A_wl)

    return chi_wl, A_wl


def gauss_seidel(T, H, pd_tol=1e-10):
    lam_min_H = np.linalg.eigvalsh(H)[0]
    tol = pd_tol * max(1.0, np.linalg.norm(H, ord=2))

    if lam_min_H <= tol:
        return None

    X = np.linalg.solve(H, T.T)
    M = T @ X
    M = 0.5 * (M + M.T)

    mu_gs = np.max(np.linalg.eigvalsh(M))
    gamma_gs = mu_gs / 2
    chi_gs = mu_gs / (1 + mu_gs)

    return chi_gs, gamma_gs


def compute_bounds(H, a, r, A, F, b):
    p = len(a)

    l = np.zeros(p)
    h = np.zeros(p)

    for i in range(p):
        lower_i = -a[i]
        upper_i = -a[i]

        for j in A:
            lower_i += H[i, j] * b[j]
            upper_i += H[i, j] * b[j]

        for j in F:
            if H[i, j] < 0:
                lower_i += H[i, j] * r[j]
            elif H[i, j] > 0:
                upper_i += H[i, j] * r[j]

        l[i] = lower_i
        h[i] = upper_i

    return l, h


def endpoint_screening(H, a, tau, r):
    p = len(a)

    A = set()
    F = set(range(p))
    b = {}

    l = np.zeros(p)
    h = np.zeros(p)

    while True:

        l, h = compute_bounds(H, a, r, A, F, b)
        new_A = {}

        for i in F:
            if l[i] > 0:
                new_A[i] = 0.0

            elif h[i] < 0:
                new_A[i] = r[i]

        if not new_A:
            break

        for i, endpoint in new_A.items():
            A.add(i)
            F.remove(i)
            b[i] = endpoint

    l, h = compute_bounds(H, a, r, A, F, b)
    m = np.zeros(p)

    for i in A:
        if b[i] == 0:
            m[i] = l[i]

        else:
            m[i] = -h[i]

    w = np.zeros(p)
    for i in range(p):
        if i in A:
            w[i] = m[i] / r[i]

    M_scr = H + 2 * np.diag(w)
    lam_min = np.linalg.eigvalsh(M_scr)[0]
    tol = 1e-10 * max(1.0, np.linalg.norm(M_scr, ord=2))

    if lam_min <= tol:
        return None

    nu = 0.99 * lam_min

    gamma_scr = tau ** 2 * min(
        1 + 2 * np.linalg.norm(np.eye(p) - H, ord=2) / nu,
        2 / nu
    )
    chi_scr = 2 * gamma_scr / (1 + 2 * gamma_scr)

    kappa = 1 + 2 * np.linalg.norm(np.eye(p) - H, ord=2) / nu

    return chi_scr, gamma_scr, kappa


def generate_M_scr(H, a, r):
    p = len(a)

    A = set()
    F = set(range(p))
    b = {}

    l = np.zeros(p)
    h = np.zeros(p)

    while True:

        l, h = compute_bounds(H, a, r, A, F, b)
        new_A = {}

        for i in F:
            if l[i] > 0:
                new_A[i] = 0.0

            elif h[i] < 0:
                new_A[i] = r[i]

        if not new_A:
            break

        for i, endpoint in new_A.items():
            A.add(i)
            F.remove(i)
            b[i] = endpoint

    l, h = compute_bounds(H, a, r, A, F, b)
    m = np.zeros(p)

    for i in A:
        if b[i] == 0:
            m[i] = l[i]

        else:
            m[i] = -h[i]

    w = np.zeros(p)
    for i in range(p):
        if i in A:
            w[i] = m[i] / r[i]

    M_scr = H + 2 * np.diag(w)

    return M_scr


def strengthen_M_scr(H, a, r, u_ref, M_old):
    r = np.asarray(r, dtype=float)
    z = np.asarray(u_ref, dtype=float)

    if not np.all(np.isfinite(z)):
        raise ValueError("Опорное решение содержит NaN или inf")

    z = np.clip(z, 0.0, r)

    g = np.array([math.fsum(H[i] * z) - a[i]for i in range(len(z))])

    eps = np.finfo(float).eps
    grad_error = 8 * eps * (np.abs(H) @ np.abs(z) + np.abs(a) + 1.0)

    terms = np.maximum((g + grad_error) * z, -(g - grad_error) * (r - z))
    delta = max(0.0, math.fsum(terms)) * (1 + 8 * eps)

    radius = np.sqrt(np.maximum(np.diag(H), 0.0) * delta)

    margin_gap = np.maximum(np.abs(g) - grad_error - radius, 0.0,)
    margin_old = np.maximum(0.5 * r * (np.diag(M_old) - np.diag(H)), 0.0,)
    margin = np.maximum(margin_old, margin_gap)
    M_new = H + 2 * np.diag(margin / r)

    diagnostics = {
        "reference_fw_gap": delta,
        "screened_old": int(np.count_nonzero(margin_old)),
        "screened_gap": int(np.count_nonzero(margin_gap)),
        "screened_combined": int(np.count_nonzero(margin)),
        "lambda_min_M_new": np.linalg.eigvalsh(M_new)[0],
    }

    return M_new, diagnostics


def screened_spectral_bound(T, H, M_scr, theta_tol=1e-6, pd_tol=1e-10):
    p = H.shape[0]

    def gamma_theta(theta):
        A = H + theta * M_scr
        X = np.linalg.solve(A, T.T)

        M = T @ X
        M = 0.5 * (M + M.T)

        lam_max = np.linalg.eigvalsh(M)[-1]

        return lam_max / (1 - theta)

    scr_tol = pd_tol * max(1.0, np.linalg.norm(M_scr, ord=2))
    lam_min_scr = np.linalg.eigvalsh(M_scr)[0]

    print(
        f"Spectral check: "
        f"lambda_min(M)={lam_min_scr:.3e}, "
        f"threshold={scr_tol:.3e}"
    )

    if lam_min_scr <= scr_tol:
        return None

    res = minimize_scalar(
        gamma_theta,
        bounds=(theta_tol, 1.0 - theta_tol),
        method="bounded",
        options={
            "xatol": 1e-8
        }
    )
    gamma_scr = 0.5 * res.fun

    lam_min_H = np.linalg.eigvalsh(H)[0]

    if lam_min_H > pd_tol:
        X0 = np.linalg.solve(H, T.T)
        M0 = T @ X0
        M0 = 0.5 * (M0 + M0.T)

        mu_gs = np.linalg.eigvalsh(M0)[-1]

        gamma_scr = min(
            gamma_scr,
            mu_gs / 2.0
        )

    chi_scr = (
        2.0 * gamma_scr
        / (1.0 + 2.0 * gamma_scr)
    )

    lam_min = np.linalg.eigvalsh(M_scr)[0]
    nu = 0.99 * lam_min
    kappa = 1.0 + 2.0 * np.linalg.norm(np.eye(p) - H, ord=2) / nu

    return chi_scr, gamma_scr, kappa


def compute_all_estimates(Q, C, u_ref=None, pd_tol=1e-10):
    H, a, T, tau, r = diag_scaling(Q, C)

    M_old = generate_M_scr(H, a, r)
    old_scr_result = screened_spectral_bound(T, H, M_old, pd_tol=pd_tol)

    scr_result = old_scr_result
    screening_diagnostics = None

    if u_ref is not None:
        M_new, screening_diagnostics = strengthen_M_scr(H, a, r, u_ref, M_old)

        new_scr_result = screened_spectral_bound(T, H, M_new, pd_tol=pd_tol)

        if new_scr_result is not None:
            if (
                scr_result is None
                or new_scr_result[0] < scr_result[0]
            ):
                scr_result = new_scr_result

    if scr_result is None:
        chi_scr = None
        gamma_scr = None
        kappa = None
    else:
        chi_scr, gamma_scr, kappa = scr_result

    if kappa is None:
        chi_wl = None
        gamma_WL = None
    else:
        chi_wl, gamma_WL = wang_lin(H, kappa)

    gs_result = gauss_seidel(T, H)

    if gs_result is None:
        chi_gs = None
        gamma_gs = None
    else:
        chi_gs, gamma_gs = gs_result

    return {
        "chi_WL": chi_wl,
        "gamma_WL": gamma_WL,
        "chi_GS": chi_gs,
        "gamma_GS": gamma_gs,
        "chi_SCR": chi_scr,
        "gamma_SCR": gamma_scr,
        "chi_SCR_old": None if old_scr_result is None else old_scr_result[0],
        "screening_diagnostics": screening_diagnostics,
    }
