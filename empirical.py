import numpy as np
from cyclic_cd import one_cycle_cd, one_cycle_cd_d
from scipy.linalg import solve_triangular, eigh
from scipy.optimize import minimize


def gap_from_reference(u, u_star, H, g_star):
    d = u - u_star
    return g_star @ d + 0.5 * d @ H @ d


def contraction_ref(u0, u_star, H, a, r, g_star):
    gap0 = gap_from_reference(u0, u_star, H, g_star)

    if gap0 <= 1e-14:
        return np.nan

    u1 = one_cycle_cd(u0, H, a, r)

    gap1 = gap_from_reference(u1, u_star, H, g_star)

    if gap1 < -1e-8:
        return np.nan

    gap1 = max(gap1, 0.0)

    return gap1 / gap0


def sample_inside(T, H, z, r):
    p = H.shape[0]

    X = np.linalg.solve(H, T.T)
    M = T @ X

    M = 0.5 * (M + M.T)

    _, eigvecs = eigh(M)
    s = eigvecs[:, -1]

    d = np.linalg.solve(H, -(np.eye(p) + T) @ s)

    t_max = np.inf
    for i in range(len(z)):
        if d[i] > 0:
            t_max = min(t_max, (r[i] - z[i]) / d[i])

        elif d[i] < 0:
            t_max = min(t_max, z[i] / (-d[i]))

    return d, t_max


def relax(u0, H, a, r, fixed_idx, n_sweeps=2):
    u = u0.copy()
    p = len(u)

    g = H @ u - a

    for _ in range(n_sweeps):
        for j in range(p):

            if j == fixed_idx:
                continue

            old = u[j]

            new = np.clip(old - g[j], 0.0, r[j])

            delta = new - old

            if delta != 0.0:
                u[j] = new
                g += delta * H[:, j]

    return u


def sample_boundary(H, T, a, r, u_star, active_tol=1e-6, n_weak=10, n_sweeps=2):
    p = len(u_star)
    g_star = H @ u_star - a

    tol = active_tol * np.maximum(r, 1.0)

    candidates = []

    for i in range(p):

        near_lower = u_star[i] <= tol[i]
        near_upper = r[i] - u_star[i] <= tol[i]

        if near_lower:
            candidates.append(
                (abs(g_star[i]), i, "lower")
            )

        elif near_upper:
            candidates.append(
                (abs(g_star[i]), i, "upper")
            )

    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:n_weak]

    starts = []

    for grad_abs, i, side in candidates:

        u = u_star.copy()

        if side == "lower":
            u[i] = r[i]
        else:
            u[i] = 0.0

        u = relax(u, H, a, r, fixed_idx=i, n_sweeps=n_sweeps)

        starts.append(u)

    return np.array(starts)


def generate_points(H, T, a, r, u_star, active_tol=1e-6):
    starts = []

    random_starts = sample_random(r, 10)

    for start in random_starts:
        starts.append(start)

        start = one_cycle_cd(start, H, a, r)
        starts.append(start)
        start = one_cycle_cd(start, H, a, r)
        starts.append(start)

    tol = active_tol * np.maximum(r, 1.0)
    is_inside = np.all((u_star > tol) & (u_star < r - tol))
    lam_min = np.linalg.eigvalsh(H)[0]
    is_pd = lam_min > 1e-10

    if is_inside and is_pd:
        d, t_max = sample_inside(T, H, u_star, r)

        for frac in [0.5, 0.1, 0.03, 0.01, 0.003, 0.001]:
            starts.append(u_star + frac * t_max * d)
    else:
        starts.extend(sample_boundary(H, T, a, r, u_star))

    return starts


def sample_random(r, n=10, seed=42):
    rng = np.random.default_rng(seed)

    return rng.uniform(
        low=0.0,
        high=r,
        size=(n, len(r))
    )


def R_and_grad(u0, u_star, H, a, r, g_star, tol=1e-12, gap_tol=1e-10):
    p = len(u0)

    u_after, d = one_cycle_cd_d(u0, H, a, r, tol)

    gap_before = gap_from_reference(u0, u_star, H, g_star)

    gap_after = gap_from_reference(u_after, u_star, H, g_star)

    if gap_before <= gap_tol:
        return np.nan, None

    if gap_after < -1e-10:
        return np.nan, None

    gap_after = max(gap_after, 0.0)
    R = gap_after / gap_before
    w = H @ u_after - a

    for i in range(p - 1, -1, -1):
        wi = w[i]

        if d[i] != 0:
            w -= H[i, :] * wi

        w[i] = 0.0

    grad_numerator = w
    grad_denominator = H @ u0 - a

    grad_R = (
        gap_before * grad_numerator
        - gap_after * grad_denominator
    ) / gap_before**2

    return R, grad_R


def select_best_starts(starts, u_star, H, a, r, g_star, n_best=5):
    scored = []

    for u0 in starts:
        R, _ = R_and_grad(u0, u_star, H, a, r, g_star)

        if np.isfinite(R):
            scored.append((R, u0.copy()))

    scored.sort(key=lambda x: x[0], reverse=True)

    return scored[:n_best]


def maximize_R(u0, u_star, H, a, r, g_star, maxiter=50):

    def func_and_grad(u):
        R, grad_R = R_and_grad(u, u_star, H, a, r, g_star)

        if not np.isfinite(R) or grad_R is None:
            return 1e6, np.zeros_like(u)

        return -R, -grad_R

    bounds = [(0.0, r[i])for i in range(len(r))]

    res = minimize(
        func_and_grad,
        u0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "ftol": 1e-9,
            "gtol": 1e-6,
            "maxls": 20,
        }
    )

    R_final, _ = R_and_grad(res.x, u_star, H, a, r, g_star)

    return R_final, res.x, res


def sample_beta(r, n=30, seed=42):
    rng = np.random.default_rng(seed)
    return r * rng.beta(0.25, 0.25, size=(n, len(r)))


def evaluate_extra(H, a, r, u_star, g_star, x):
    h_scale = max(1.0, np.linalg.norm(H, np.inf))

    delta = x - u_star
    gap = g_star @ delta + 0.5 * delta @ H @ delta

    gap_floor = max(1e-10, 1e-8 * h_scale * (delta @ delta))
    if not np.isfinite(gap) or gap <= gap_floor:
        return np.nan, None

    value, gradient = R_and_grad(x, u_star, H, a, r, g_star)

    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        return np.nan, None

    if gradient is None or not np.all(np.isfinite(gradient)):
        return value, None

    return value, gradient


def experiment_worst_case(f_star, u_star, H, T, a, r, ref_residual, n_best=5, tol=1e-12):
    # if ref_residual > 1e-6:
    #     return np.nan, None
    g_star = H @ u_star - a

    starts = generate_points(H, T, a, r, u_star)

    best_starts = select_best_starts(
        starts, u_star, H, a, r, g_star, n_best=n_best)

    if not best_starts:
        return np.nan, None

    best_R = best_starts[0][0]
    best_u = best_starts[0][1]

    local_results = []

    for R0, u0 in best_starts:

        R_local, u_local, res = maximize_R(
            u0, u_star, H, a, r, g_star, maxiter=50)

        local_results.append({
            "R_start": R0,
            "R_final": R_local,
            "iterations": res.nit,
            "nfev": res.nfev,
            "success": res.success
        })

        if np.isfinite(R_local) and R_local > best_R:
            best_R = R_local
            best_u = u_local.copy()

    extra_starts = [best_u.copy()]
    extra_starts.extend(sample_beta(r))

    free = (
        (u_star > 1e-7 * np.maximum(1, r))
        & (u_star < r - 1e-7 * np.maximum(1, r))
    )
    A = H[np.ix_(free, free)]

    if np.count_nonzero(free) > 1:
        A = H[np.ix_(free, free)]
        eigenvalues = np.linalg.eigvalsh(A)
        pd_tol = max(tol, 1e-10 * max(1.0, eigenvalues[-1]))

        if eigenvalues[0] > pd_tol:
            B = -solve_triangular(np.tril(A), np.triu(A, 1), lower=True,)

            Q = B.T @ A @ B
            Q = 0.5 * (Q + Q.T)

            _, eigenvectors = eigh(Q, A, subset_by_index=[
                                   len(A) - 1, len(A) - 1],)

            direction = np.zeros_like(u_star, dtype=float)
            direction[free] = eigenvectors[:, -1]

            for sign in (-1.0, 1.0):
                d = sign * direction

                limits = np.full(len(r), np.inf)
                positive = d > 0
                negative = d < 0

                limits[positive] = (
                    (r[positive] - u_star[positive]) / d[positive]
                )
                limits[negative] = (
                    -u_star[negative] / d[negative]
                )

                t_max = np.min(limits)

                if np.isfinite(t_max) and t_max > 0:
                    for fraction in (0.8, 0.3, 0.1, 0.03):
                        point = u_star + fraction * t_max * d
                        extra_starts.append(np.clip(point, 0, r))

    h_scale = max(1.0, np.linalg.norm(H, np.inf))

    ranked = []
    for point in extra_starts:
        value, _ = evaluate_extra(H, a, r, u_star, g_star, point)
        if np.isfinite(value):
            ranked.append((value, point.copy()))

    ranked.sort(key=lambda item: item[0], reverse=True)

    for start_value, x in ranked[:n_best]:
        if start_value > best_R:
            best_R = start_value
            best_u = x.copy()

        for iteration in range(100):
            value, gradient = evaluate_extra(H, a, r, u_star, g_star, x)
            if gradient is None:
                break

            gradient_q = r * gradient
            norm = np.linalg.norm(gradient_q, np.inf)

            if norm == 0:
                break

            direction_q = gradient_q / norm
            accepted = False

            for j in range(24):
                step = 0.5 ** j

                trial = r * np.clip(x / r + step * direction_q, 0.0, 1.0,)

                displacement = trial - x
                predicted_gain = gradient @ displacement

                if predicted_gain <= 0:
                    continue

                trial_value, _ = evaluate_extra(H, a, r, u_star, g_star, trial)

                if np.isfinite(trial_value) and trial_value > value + 1e-4 * predicted_gain:
                    x = trial
                    accepted = True

                    if trial_value > best_R:
                        best_R = trial_value
                        best_u = x.copy()

                    break

            if not accepted:
                break

    return best_R, best_u
