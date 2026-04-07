import numpy as np
from ar_models import make_ar_design


def simulate_ar(nobs, intercept, phi, sigma=1.0, burn=500, seed=123):
    """Simulate an AR process with nonnegative intercept and AR coefficients."""
    phi = np.asarray(phi, dtype=float)
    p = len(phi)

    if intercept < 0.0:
        raise ValueError("intercept must be nonnegative")
    if np.any(phi < 0.0):
        raise ValueError("all ar coefficients must be nonnegative")
    if phi.sum() >= 1.0:
        raise ValueError("for nonnegative phi, require sum(phi) < 1 for stationarity")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    rng = np.random.default_rng(seed)
    n_total = nobs + burn
    y = np.empty(n_total)

    mu = intercept / (1.0 - phi.sum())
    y[:p] = mu

    eps = rng.normal(loc=0.0, scale=sigma, size=n_total)

    for t in range(p, n_total):
        y[t] = intercept + np.dot(phi, y[t-p:t][::-1]) + eps[t]

    return y[burn:]


def evaluate_closeness(fit, intercept_true, phi_true, y):
    """Evaluate closeness of a fitted AR model to the true model."""
    phi_true = np.asarray(phi_true, dtype=float)
    p = len(phi_true)
    beta_true = np.r_[intercept_true, phi_true]
    beta_hat = fit["beta"]

    beta_err = beta_hat - beta_true
    phi_err = fit["phi"] - phi_true

    _, x = make_ar_design(y, p)
    fitted_true = x @ beta_true
    fitted_hat = fit["fitted"]
    fitted_err = fitted_hat - fitted_true

    out = {
        "intercept_error": fit["intercept"] - intercept_true,
        "intercept_abs_error": abs(fit["intercept"] - intercept_true),
        "phi_mae": np.mean(np.abs(phi_err)),
        "phi_rmse": np.sqrt(np.mean(phi_err**2)),
        "phi_max_abs_error": np.max(np.abs(phi_err)),
        "beta_mae": np.mean(np.abs(beta_err)),
        "beta_rmse": np.sqrt(np.mean(beta_err**2)),
        "beta_l2_error": np.linalg.norm(beta_err),
        "beta_max_abs_error": np.max(np.abs(beta_err)),
        "cond_mean_mae": np.mean(np.abs(fitted_err)),
        "cond_mean_rmse": np.sqrt(np.mean(fitted_err**2)),
        "cond_mean_max_abs_error": np.max(np.abs(fitted_err)),
    }

    sum_phi_true = phi_true.sum()
    sum_phi_hat = fit["phi"].sum()

    out["long_run_mean_true"] = intercept_true / (1.0 - sum_phi_true)

    if sum_phi_hat < 1.0:
        out["long_run_mean_hat"] = fit["intercept"] / (1.0 - sum_phi_hat)
        out["long_run_mean_error"] = out["long_run_mean_hat"] - out["long_run_mean_true"]
        out["long_run_mean_abs_error"] = abs(out["long_run_mean_error"])
    else:
        out["long_run_mean_hat"] = np.nan
        out["long_run_mean_error"] = np.nan
        out["long_run_mean_abs_error"] = np.nan

    return out


def print_closeness(name, ev):
    """Print closeness of fitted model to true model."""
    print(name)
    print(f"  intercept_error          = {ev['intercept_error']:.6f}")
    print(f"  intercept_abs_error      = {ev['intercept_abs_error']:.6f}")
    print(f"  phi_mae                  = {ev['phi_mae']:.6f}")
    print(f"  phi_rmse                 = {ev['phi_rmse']:.6f}")
    print(f"  phi_max_abs_error        = {ev['phi_max_abs_error']:.6f}")
    print(f"  beta_mae                 = {ev['beta_mae']:.6f}")
    print(f"  beta_rmse                = {ev['beta_rmse']:.6f}")
    print(f"  beta_l2_error            = {ev['beta_l2_error']:.6f}")
    print(f"  beta_max_abs_error       = {ev['beta_max_abs_error']:.6f}")
    print(f"  cond_mean_mae            = {ev['cond_mean_mae']:.6f}")
    print(f"  cond_mean_rmse           = {ev['cond_mean_rmse']:.6f}")
    print(f"  cond_mean_max_abs_error  = {ev['cond_mean_max_abs_error']:.6f}")
    print(f"  long_run_mean_hat        = {ev['long_run_mean_hat']:.6f}")
    print(f"  long_run_mean_true       = {ev['long_run_mean_true']:.6f}")
    print(f"  long_run_mean_error      = {ev['long_run_mean_error']:.6f}")
    print(f"  long_run_mean_abs_error  = {ev['long_run_mean_abs_error']:.6f}")
    print()


def better_name(metric, eval_ols, eval_nn, eval_desc):
    """Return the name of the fit with the smallest value of a metric."""
    vals = {
        "ols": eval_ols[metric],
        "nonnegative": eval_nn[metric],
        "descending": eval_desc[metric],
    }
    return min(vals, key=vals.get)
