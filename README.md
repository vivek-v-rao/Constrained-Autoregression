# Constrained-Autoregression

Python code for fitting univariate autoregressive models with sign and shape constraints.

The project includes:

- ordinary least-squares AR(p) estimation
- AR(p) estimation with non-negative coefficients
- optional non-negative intercept
- optional monotone non-increasing AR coefficients by lag
- simulation of constrained AR processes
- in-sample comparison with a known data-generating process
- out-of-sample forecast comparison
- fitting constrained AR models to user-supplied data files

## Files

- `ar_models.py`  
  Core fitting routines for unrestricted and constrained AR models.

- `ar_sim.py`  
  Simulation and evaluation helpers.

- `xar_nonneg_demo.py`  
  Simulates data from a known AR process and compares unrestricted, non-negative, and descending-constrained fits.

- `xar_nonneg_oos.py`  
  Simulates data, splits the sample into training and test sets, compares out-of-sample one-step-ahead forecast errors, and can write the simulated series to a text file.

- `xfit_ar_data.py`  
  Fits unrestricted and constrained AR models to a user-supplied data file.

## Requirements

- Python 3
- NumPy
- SciPy
- pandas

Install dependencies with:

```bash
pip install numpy scipy pandas
```

## Model

For a univariate AR(p) model

```text
y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p} + e_t
```

the project supports:

1. **Unrestricted OLS fit**
2. **Non-negative constrained fit**
   - `c >= 0` if requested
   - `phi_i >= 0` for all lags
3. **Descending constrained fit**
   - `c >= 0` if requested
   - `phi_1 >= phi_2 >= ... >= phi_p >= 0`

## Main functions

### `fit_ar_ols(y, p)`

Fits an unrestricted AR(p) by ordinary least squares.

Returns a dictionary containing:

- intercept
- AR coefficients
- fitted values
- residuals
- SSE
- estimated innovation variance
- log-likelihood
- AIC
- BIC

### `fit_ar_nonnegative(y, p, nonnegative_intercept=True, descending=False)`

Fits a constrained AR(p).

Arguments:

- `y`: one-dimensional data array
- `p`: AR order
- `nonnegative_intercept`: if `True`, constrain the intercept to be non-negative
- `descending`: if `True`, impose monotone non-increasing AR coefficients by lag

Implementation:

- if `descending=False`, the model is fit with simple non-negativity bounds using `scipy.optimize.lsq_linear`
- if `descending=True`, the model is fit with linear inequality constraints using `scipy.optimize.minimize(..., method="trust-constr")`

## Usage

### 1. Simulated example

Run:

```bash
python xar_nonneg_demo.py
```

This script:

- simulates data from a non-negative AR process
- fits unrestricted, non-negative, and descending-constrained AR models
- compares estimated coefficients with the true parameters
- reports closeness to the true model

### 2. Out-of-sample forecasting example

Run:

```bash
python xar_nonneg_oos.py
```

This script:

- simulates a series
- splits the sample into training and test subsamples
- fits the AR models on the training sample
- compares one-step-ahead out-of-sample forecast errors on the test sample

At present, `xar_nonneg_oos.py` controls simulated-data output with the variable:

```python
out_sim_data_file = "sim_data.txt"  # or None
```

When that variable is not `None`, the script writes the simulated series to a plain text file with one numeric value per line and no header.

### 3. Fit a constrained AR model to your own data

Run:

```bash
python xfit_ar_data.py mydata.txt --p 10
```

By default, `xfit_ar_data.py` assumes the input file has **no header line**.

If the file has a header row, use:

```bash
python xfit_ar_data.py mydata.csv --p 10 --header
```

To allow a negative intercept:

```bash
python xfit_ar_data.py mydata.txt --p 10 --allow-negative-intercept
```

To impose monotone non-increasing coefficients:

```bash
python xfit_ar_data.py mydata.txt --p 10 --descending
```

### Selecting a column

If the input file has multiple columns and you want a specific one:

```bash
python xfit_ar_data.py mydata.csv --p 10 --header --column y
```

If no column is specified, the first column is used.

## Notes

- `simulate_ar()` assumes non-negative coefficients and requires `sum(phi) < 1` for stationarity.
- The descending constraint is stronger than simple non-negativity.
- When `descending=True`, exact gradient and Hessian information are supplied to the optimizer for the quadratic objective.
- For simple numeric text files with one value per line, `xfit_ar_data.py` uses `numpy.loadtxt` by default when no header is assumed.

## Possible extensions

Natural next steps include:

- automatic lag-order selection
- rolling or expanding window forecasts
- multi-step forecasting
- comparison with `statsmodels`
- support for additional linear constraints
- packaging as an installable module

## License

Add the license of your choice, for example MIT.
