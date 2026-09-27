import numpy as np
from scipy.optimize import minimize


def one_cycle_cd(u0, H, a, r):
    u = u0.copy()
    g = H @ u - a

    for i in range(len(u)):
        old = u[i]

        new = np.clip(
            old - g[i],
            0.0,
            r[i]
        )

        delta = new - old

        if delta != 0.0:
            u[i] = new
            g += delta * H[:, i]

    return u


def one_cycle_cd_d(u0, H, a, r, tol=1e-12):
    u = u0.copy()
    g = H @ u - a

    d = np.zeros(len(u))

    for i in range(len(u)):
        old = u[i]

        new = np.clip(old - g[i], 0.0, r[i])

        delta = new - old

        if tol < old - g[i] < r[i] - tol:
            d[i] = 1.0
        else:
            d[i] = 0

        if delta != 0.0:
            u[i] = new
            g += delta * H[:, i]

    return u, d


def solve(H, a, r):
    p = len(a)

    def f(u):
        return u.T @ H @ u / 2 - a.T @ u

    def g(u):
        return H @ u - a.T

    result = minimize(
        f,
        x0=r / 2,
        jac=g,
        bounds=[(0.0, r[i]) for i in range(p)],
        method="L-BFGS-B",
        options={
            "ftol": 1e-14,
            "gtol": 1e-12,
            "maxiter": 10000
        }
    )

    if not result.success:
        print("warning:", result.message)

    return result.x, result.fun


def solve_cd(H, a, r, tol=1e-8, max_cycles=5000):
    u = r / 2

    for _ in range(max_cycles):
        u = one_cycle_cd(u, H, a, r)

        g = H @ u - a
        projected = np.clip(u - g, 0.0, r)
        residual = np.linalg.norm(u - projected)

        if residual <= tol:
            return u, objective(u, H, a), residual

    return u, objective(u, H, a), residual


def objective(u, H, a):
    return 0.5 * u @ H @ u - a @ u


def residual_norm(u, H, a, r):
    g = H @ u - a
    projected = np.clip(u - g, 0.0, r)

    return np.linalg.norm(u - projected)
