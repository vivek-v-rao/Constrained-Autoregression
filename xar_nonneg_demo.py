import numpy as np
import pandas as pd
from ar_models import fit_ar_ols, fit_ar_nonnegative, print_fit
from ar_sim import simulate_ar, evaluate_closeness, print_closeness, better_name


def coef_dataframe(intercept_true, phi_true, fit_ols, fit_nn, fit_desc=None):
    """Return dataframe of true and fitted intercept/AR coefficients."""
    phi_true = np.asarray(phi_true, dtype=float)

    index = ["intercept"] + [f"phi_{i}" for i in range(1, len(phi_true) + 1)]

    data = {
        "true": np.r_[intercept_true, phi_true],
        "unconstrained": fit_ols["beta"],
        "constrained": fit_nn["beta"],
    }

    if fit_desc is not None:
        data["descending"] = fit_desc["beta"]

    return pd.DataFrame(data, index=index)


def main():
    nobs = 100
    intercept_true = 0.4
    p_fit = 15
    phi_true = 0.05 * np.ones(p_fit)
    sigma_true = 1.0

    y = simulate_ar(
        nobs=nobs,
        intercept=intercept_true,
        phi=phi_true,
        sigma=sigma_true,
        burn=500,
        seed=12345,
    )

    fit_ols = fit_ar_ols(y, p_fit)
    fit_nn = fit_ar_nonnegative(y, p_fit, nonnegative_intercept=True, descending=False)
    fit_desc = fit_ar_nonnegative(y, p_fit, nonnegative_intercept=True, descending=True)

    eval_ols = evaluate_closeness(fit_ols, intercept_true, phi_true, y)
    eval_nn = evaluate_closeness(fit_nn, intercept_true, phi_true, y)
    eval_desc = evaluate_closeness(fit_desc, intercept_true, phi_true, y)

    df_coef = coef_dataframe(intercept_true, phi_true, fit_ols, fit_nn, fit_desc)

    print("true model")
    print(f"  intercept = {intercept_true:.6f}")
    print(f"  phi       = {np.array2string(phi_true, precision=6)}")
    print(f"  sum(phi)  = {phi_true.sum():.6f}")
    print()

    print_fit("unrestricted ols fit", fit_ols)
    print_fit("nonnegative constrained fit", fit_nn)
    print_fit("nonnegative descending fit", fit_desc)

    print("coefficient comparison")
    print(df_coef.round(6))
    print()

    print_closeness("closeness of unrestricted ols fit to true model", eval_ols)
    print_closeness("closeness of nonnegative constrained fit to true model", eval_nn)
    print_closeness("closeness of nonnegative descending fit to true model", eval_desc)

    print("best fit by metric")
    print(f"  beta_rmse               = {better_name('beta_rmse', eval_ols, eval_nn, eval_desc)}")
    print(f"  phi_rmse                = {better_name('phi_rmse', eval_ols, eval_nn, eval_desc)}")
    print(f"  cond_mean_rmse          = {better_name('cond_mean_rmse', eval_ols, eval_nn, eval_desc)}")
    print(f"  long_run_mean_abs_error = {better_name('long_run_mean_abs_error', eval_ols, eval_nn, eval_desc)}")


if __name__ == "__main__":
    main()
