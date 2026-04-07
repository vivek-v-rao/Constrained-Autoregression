import argparse
import numpy as np
import pandas as pd
from ar_models import fit_ar_ols, fit_ar_nonnegative, print_fit


def read_series(infile, column=None, header=False):
    """Read a univariate series from a text or csv file."""
    if column is None and not header:
        y = np.loadtxt(infile, comments="#", ndmin=1)
        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            return y, "first column"

        if y.ndim == 2:
            return y[:, 0], "first column"

        raise ValueError("input data must be 1d or 2d")

    df = pd.read_csv(
        infile,
        header=0 if header else None,
        sep=r"\s+|,",
        engine="python",
        comment="#",
    )

    if column is None:
        s = df.iloc[:, 0]
        column_name = "first column"
    else:
        s = df[column]
        column_name = str(column)

    y = pd.to_numeric(s, errors="coerce").dropna().to_numpy()

    return y, column_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("--column", default=None)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument(
        "--header",
        action="store_true",
        help="treat first line of input file as a header",
    )
    parser.add_argument("--allow-negative-intercept", action="store_true")
    parser.add_argument("--descending", action="store_true")
    args = parser.parse_args()

    y, column_name = read_series(
        args.infile,
        column=args.column,
        header=args.header,
    )

    fit_ols = fit_ar_ols(y, args.p)
    fit_nn = fit_ar_nonnegative(
        y,
        args.p,
        nonnegative_intercept=not args.allow_negative_intercept,
        descending=args.descending,
    )

    print(f"nobs read = {len(y)}")
    print(f"p         = {args.p}")
    print(f"column    = {column_name}")
    print(f"header    = {args.header}")
    print()

    print_fit("unrestricted ols fit", fit_ols)
    print_fit("constrained fit", fit_nn)

    print("unrestricted ols estimate")
    print(f"  beta_ols  = {np.array2string(fit_nn['beta_ols_start'], precision=6)}")


if __name__ == "__main__":
    main()
