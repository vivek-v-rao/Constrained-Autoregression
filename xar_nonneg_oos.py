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


def one_step_forecast_errors(y, fit, n_train):
    """Return out-of-sample one-step-ahead forecast errors using actual lagged values."""
    p = fit["p"]
    beta = fit["beta"]
    intercept = beta[0]
    phi = beta[1:]

    if n_train < p:
        raise ValueError("n_train must be at least p")
    if len(y) <= n_train:
        raise ValueError("need at least one out-of-sample observation")

    y = np.asarray(y, dtype=float)
    yhat = np.empty(len(y) - n_train)
    err = np.empty(len(y) - n_train)

    for j, t in enumerate(range(n_train, len(y))):
        lags = y[t - p:t][::-1]
        yhat[j] = intercept + np.dot(phi, lags)
        err[j] = y[t] - yhat[j]

    return {
        "y_true": y[n_train:],
        "y_pred": yhat,
        "errors": err,
        "rmse": np.sqrt(np.mean(err**2)),
        "mae": np.mean(np.abs(err)),
        "mse": np.mean(err**2),
        "mean_error": np.mean(err),
        "max_abs_error": np.max(np.abs(err)),
    }


def forecast_error_dataframe(fc_ols, fc_nn, fc_desc):
    """Return dataframe comparing out-of-sample forecast errors."""
    return pd.DataFrame(
        {
            "unconstrained": [
                fc_ols["rmse"],
                fc_ols["mae"],
                fc_ols["mse"],
                fc_ols["mean_error"],
                fc_ols["max_abs_error"],
            ],
            "constrained": [
                fc_nn["rmse"],
                fc_nn["mae"],
                fc_nn["mse"],
                fc_nn["mean_error"],
                fc_nn["max_abs_error"],
            ],
            "descending": [
                fc_desc["rmse"],
                fc_desc["mae"],
                fc_desc["mse"],
                fc_desc["mean_error"],
                fc_desc["max_abs_error"],
            ],
        },
        index=["rmse", "mae", "mse", "mean_error", "max_abs_error"],
    )


def write_simulated_data(y, out_sim_data_file):
    """Write simulated data values only, one per line, if output file is supplied."""
    if out_sim_data_file is None:
        return

    np.savetxt(out_sim_data_file, np.asarray(y, dtype=float), fmt="%.18g")


def main():
    nobs = 300
    train_frac = 0.67
    intercept_true = 0.4
    p_fit = 15
    phi_true = 0.05 * np.ones(p_fit)
    sigma_true = 1.0
    out_sim_data_file = "sim_data.txt" # None
    # out_sim_data_file = "sim_data.txt"

    y = simulate_ar(
        nobs=nobs,
        intercept=intercept_true,
        phi=phi_true,
        sigma=sigma_true,
        burn=500,
        seed=12345,
    )

    n_train = int(train_frac * nobs)
    if n_train <= p_fit:
        raise ValueError("training sample must be longer than p_fit")

    y_train = y[:n_train]
    y_test = y[n_train:]

    write_simulated_data(y, out_sim_data_file)

    fit_ols = fit_ar_ols(y_train, p_fit)
    fit_nn = fit_ar_nonnegative(
        y_train,
        p_fit,
        nonnegative_intercept=True,
        descending=False,
    )
    fit_desc = fit_ar_nonnegative(
        y_train,
        p_fit,
        nonnegative_intercept=True,
        descending=True,
    )

    eval_ols = evaluate_closeness(fit_ols, intercept_true, phi_true, y_train)
    eval_nn = evaluate_closeness(fit_nn, intercept_true, phi_true, y_train)
    eval_desc = evaluate_closeness(fit_desc, intercept_true, phi_true, y_train)

    fc_ols = one_step_forecast_errors(y, fit_ols, n_train)
    fc_nn = one_step_forecast_errors(y, fit_nn, n_train)
    fc_desc = one_step_forecast_errors(y, fit_desc, n_train)

    df_coef = coef_dataframe(intercept_true, phi_true, fit_ols, fit_nn, fit_desc)
    df_fc = forecast_error_dataframe(fc_ols, fc_nn, fc_desc)

    print("true model")
    print(f"  intercept = {intercept_true:.6f}")
    print(f"  phi       = {np.array2string(phi_true, precision=6)}")
    print(f"  sum(phi)  = {phi_true.sum():.6f}")
    print()

    print("sample split")
    print(f"  total observations      = {len(y)}")
    print(f"  training observations   = {len(y_train)}")
    print(f"  out-of-sample obs       = {len(y_test)}")
    print()

    if out_sim_data_file is not None:
        print(f"  wrote simulated data    = {out_sim_data_file}")
        print()

    print_fit("unrestricted ols fit on training sample", fit_ols)
    print_fit("nonnegative constrained fit on training sample", fit_nn)
    print_fit("nonnegative descending fit on training sample", fit_desc)

    print("coefficient comparison")
    print(df_coef.round(6))
    print()

    print_closeness("closeness of unrestricted ols fit to true model", eval_ols)
    print_closeness("closeness of nonnegative constrained fit to true model", eval_nn)
    print_closeness("closeness of nonnegative descending fit to true model", eval_desc)

    print("out-of-sample one-step-ahead forecast error")
    print(df_fc.round(6))
    print()

    print("best fit by metric")
    print(f"  in-sample beta_rmse     = {better_name('beta_rmse', eval_ols, eval_nn, eval_desc)}")
    print(f"  in-sample phi_rmse      = {better_name('phi_rmse', eval_ols, eval_nn, eval_desc)}")
    print(f"  out-of-sample rmse      = {df_fc.loc['rmse'].astype(float).idxmin()}")
    print(f"  out-of-sample mae       = {df_fc.loc['mae'].astype(float).idxmin()}")
    print(f"  out-of-sample max_abs   = {df_fc.loc['max_abs_error'].astype(float).idxmin()}")


if __name__ == "__main__":
    main()
