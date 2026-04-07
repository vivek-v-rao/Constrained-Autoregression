import numpy as np
from scipy.optimize import lsq_linear, minimize, Bounds, LinearConstraint


def make_ar_design(y, p):
    """Return response and design matrix for AR(p) with intercept."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional")
    if len(y) <= p:
        raise ValueError("len(y) must be greater than p")
    if p < 1:
        raise ValueError("p must be at least 1")

    yt = y[p:]
    xlags = np.column_stack([y[p - k - 1:len(y) - k - 1] for k in range(p)])
    x = np.column_stack([np.ones(len(yt)), xlags])
    return yt, x


def calc_fit_stats(resid, npar):
    """Return SSE, sigma^2, loglik, AIC, and BIC from residuals."""
    nobs = len(resid)
    sse = np.dot(resid, resid)
    sigma2 = sse / nobs

    if sigma2 <= 0.0:
        loglik = np.inf
        aic = -np.inf
        bic = -np.inf
    else:
        loglik = -0.5 * nobs * (np.log(2.0 * np.pi) + 1.0 + np.log(sigma2))
        aic = 2.0 * npar - 2.0 * loglik
        bic = np.log(nobs) * npar - 2.0 * loglik

    return {
        "sse": sse,
        "sigma2": sigma2,
        "loglik": loglik,
        "aic": aic,
        "bic": bic,
    }


def fit_ar_ols(y, p):
    """Fit AR(p) by ordinary least squares."""
    yt, x = make_ar_design(y, p)
    beta, _, _, _ = np.linalg.lstsq(x, yt, rcond=None)
    fitted = x @ beta
    resid = yt - fitted
    stats = calc_fit_stats(resid, npar=p + 1)

    return {
        "intercept": beta[0],
        "phi": beta[1:],
        "beta": beta,
        "fitted": fitted,
        "residuals": resid,
        "sse": stats["sse"],
        "sigma2": stats["sigma2"],
        "loglik": stats["loglik"],
        "aic": stats["aic"],
        "bic": stats["bic"],
        "success": True,
        "message": "ordinary least squares",
        "p": p,
        "nobs_fit": len(yt),
        "npar": p + 1,
    }


def fit_ar_nonnegative(y, p, nonnegative_intercept=True, descending=False):
    """Fit AR(p) with nonnegative coefficients and optional monotone decline by lag."""
    yt, x = make_ar_design(y, p)

    beta_ols, _, _, _ = np.linalg.lstsq(x, yt, rcond=None)

    lb = np.zeros(p + 1)
    if not nonnegative_intercept:
        lb[0] = -np.inf
    ub = np.full(p + 1, np.inf)

    if not descending:
        res = lsq_linear(x, yt, bounds=(lb, ub))
        beta = res.x
        success = res.success
        message = res.message
    else:
        x0 = beta_ols.copy()
        if nonnegative_intercept:
            x0[0] = max(x0[0], 0.0)
        x0[1:] = np.maximum(x0[1:], 0.0)

        constraints = []
        if p > 1:
            a = np.zeros((p - 1, p + 1))
            for i in range(p - 1):
                a[i, i + 1] = 1.0
                a[i, i + 2] = -1.0
            constraints.append(
                LinearConstraint(a, np.zeros(p - 1), np.full(p - 1, np.inf))
            )

        xtx = x.T @ x
        xty = x.T @ yt
        yty = yt @ yt

        def obj(beta):
            return 0.5 * beta @ xtx @ beta - beta @ xty + 0.5 * yty

        def jac(beta):
            return xtx @ beta - xty

        def hess(beta):
            return xtx

        res = minimize(
            obj,
            x0=x0,
            method="trust-constr",
            jac=jac,
            hess=hess,
            bounds=Bounds(lb, ub),
            constraints=constraints,
        )
        beta = res.x
        success = res.success
        message = res.message

    fitted = x @ beta
    resid = yt - fitted
    stats = calc_fit_stats(resid, npar=p + 1)

    return {
        "intercept": beta[0],
        "phi": beta[1:],
        "beta": beta,
        "fitted": fitted,
        "residuals": resid,
        "sse": stats["sse"],
        "sigma2": stats["sigma2"],
        "loglik": stats["loglik"],
        "aic": stats["aic"],
        "bic": stats["bic"],
        "success": success,
        "message": message,
        "beta_ols_start": beta_ols,
        "p": p,
        "nobs_fit": len(yt),
        "npar": p + 1,
        "nonnegative_intercept": nonnegative_intercept,
        "descending": descending,
    }


def print_fit(name, fit):
    """Print a fitted AR model."""
    print(name)
    print(f"  intercept = {fit['intercept']:.6f}")
    print(f"  phi       = {np.array2string(fit['phi'], precision=6)}")
    print(f"  sum(phi)  = {fit['phi'].sum():.6f}")
    print(f"  sse       = {fit['sse']:.6f}")
    print(f"  sigma2    = {fit['sigma2']:.6f}")
    print(f"  loglik    = {fit['loglik']:.6f}")
    print(f"  aic       = {fit['aic']:.6f}")
    print(f"  bic       = {fit['bic']:.6f}")
    print(f"  success   = {fit['success']}")
    print(f"  message   = {fit['message']}")
    print()
