# This module contains all the functions to run the stock simulator
import numpy as np
import pandas as pd
from arch import arch_model

def wiener_process(n_steps = 252, T=1, n_paths = 5):
    '''
    This functioin simulates a Wiener process (Brownian motion) with set parameters
    parameters:
    - n_steps: number of time steps in the simulation (default: 252, representing trading days in a year)
    - T: total time duration of the simulation (default: 1 year)
    - n_paths: number of independent paths to simulate (default: 5)
    Returns:
    - W: a 2D array of shape (n_steps, n_paths) containing the simulated Wiener process values
    - times: a 1D array of time points corresponding to the steps
    '''
    times = np.linspace(0,T,n_steps)
    np.random.seed(42)
    dt = T/n_steps
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=(n_steps, n_paths))
    W = np.cumsum(dW, axis=0)
    return W, times

def simulate_gbm(S0=100.0, mu=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=1, seed=None):
    """
    Simulate Geometric Brownian Motion using the exact solution.

    Parameters
    ----------
    S0 : float, initial value (e.g., stock price)
    mu : float, drift (expected return per unit time)
    sigma : float, volatility per unit time
    T : float, total time horizon
    n_steps : int, number of time steps
    n_paths : int, number of independent paths
    seed : int or None, random seed

    Returns
    -------
    df : pandas.DataFrame
        Index = time points, columns = path_0, path_1, ..., path_{n_paths-1}
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    # Standard Brownian motion increments
    dW = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)

    # Exact GBM solution
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

    # Build DataFrame: time as index, paths as columns
    df = pd.DataFrame(
        S.T,
        index=pd.Index(t, name="time"),
        columns=[f"path_{i}" for i in range(n_paths)]
    )

    return df

def calculate_moving_average(data, window_size = 20, method="simple", columns=None):
    '''Calculate a moving average for a pandas Series or DataFrame.

    Parameters:
    data: A pandas Series or DataFrame containing the price data.
    window_size: The number of periods to include in the average.
    method: Either "simple" for a simple moving average or "exponential" for an
        exponential moving average.
    columns: Optional column selection. When omitted, all columns are processed.
        Accepts a column name, a list of column names, or a tuple of column names.

    Returns:
    A pandas Series when the input is a Series, or a pandas DataFrame when the
    input is a DataFrame.
    '''
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    method_name = method.lower()
    if method_name not in {"simple", "exponential"}:
        raise ValueError("method must be either 'simple' or 'exponential'.")

    if isinstance(data, pd.Series):
        if method_name == "simple":
            return data.rolling(window=window_size).mean()
        return data.ewm(span=window_size, adjust=False).mean()

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas Series or DataFrame.")

    if columns is None:
        selected_columns = list(data.columns)
    elif isinstance(columns, str):
        selected_columns = [columns]
    else:
        selected_columns = list(columns)

    missing_columns = [column for column in selected_columns if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")

    result = pd.DataFrame(index=data.index)
    for column in selected_columns:
        series = data[column]
        if method_name == "simple":
            result[column] = series.rolling(window=window_size).mean()
        else:
            result[column] = series.ewm(span=window_size, adjust=False).mean()

    return result.dropna()

def calculate_rolling_volatility(data, window_size=20, columns=None, method = "log", annualize=True):
    """
    Calculate rolling volatility for one or more columns in a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe with price series in columns.
    window_size : int, default 20
        Number of periods used for the rolling standard deviation.
    columns : str, list-like, or None, default None
        Column(s) to process. If None, all columns are used.
    method : str, default log
        method to calculate returns used to calculate volatility
    annualize : bool, default True
        Whether to annualize volatility using sqrt(252).

    Returns
    -------
    pandas.DataFrame
        DataFrame with rolling volatility values for each selected column.
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if columns is None:
        selected_columns = list(data.columns)
    elif isinstance(columns, str):
        selected_columns = [columns]
    else:
        selected_columns = list(columns)

    missing_columns = [col for col in selected_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")

    result = pd.DataFrame(index=data.index)

    for col in selected_columns:
        if method == "log":
            log_returns = np.log(data[col]).diff().dropna()
            volatility = log_returns.rolling(window=window_size).std()
        elif method == "pct":
            pct_returns = data[col].pct_change()
            volatility = pct_returns.rolling(window=window_size).std()
        else:
            raise KeyError("Method is either 'pct' or 'log'")
        if annualize:
            volatility = volatility * np.sqrt(252)
        result[f"{col} volatiltity"] = volatility

    return result.dropna()

def simulate_garch(n_steps=252, mu=0.0003, omega=0.00001, alpha=0.1, beta=0.87,
                    start_price=100, n_paths=1, seed=42, burn=100):
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be < 1 for a stationary process")

    sim = arch_model(None, mean="constant", vol="Garch", p=1, q=1, dist="normal")
    sim.distribution.random_state = np.random.RandomState(seed)

    prices = np.empty((n_steps + 1, n_paths))
    returns = np.full((n_steps + 1, n_paths), np.nan)   # NaN, not garbage, for row 0
    true_vol = np.full((n_steps + 1, n_paths), np.nan)
    prices[0] = start_price

    for p in range(n_paths):
        simulated = sim.simulate([mu, omega, alpha, beta], nobs=n_steps, burn=burn)
        returns[1:, p] = simulated["data"].values
        true_vol[1:, p] = simulated["volatility"].values      # model's actual sigma_t
        prices[1:, p] = start_price * np.cumprod(1 + returns[1:, p])

    cols = [f"Path_{i}" for i in range(n_paths)]
    prices = pd.DataFrame(prices, columns=cols)
    returns = pd.DataFrame(returns, columns=cols).dropna()
    true_vol = pd.DataFrame(true_vol, columns=cols)  # the real conditional volatility

    return prices, true_vol, returns


def train_garch_rolling_forecast(data, initial_train_size=100, test_size=20, p=1, q=1):
    """Split returns into train/test sets, fit a GARCH model iteratively, and evaluate one-step forecasts.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        A univariate series of returns. If a DataFrame is supplied, the first column is used.
    initial_train_size : int, default 100
        Number of observations used for the first training window.
    test_size : int, default 20
        Number of one-step-ahead forecasts to evaluate.
    p : int, default 1
        GARCH autoregressive lag order.
    q : int, default 1
        GARCH moving-average lag order.

    Returns
    -------
    dict
        A dictionary with the train split, test split, forecast table, and evaluation metrics.
    """
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("data must be a univariate Series or a single-column DataFrame")
        series = data.iloc[:, 0]
    elif isinstance(data, pd.Series):
        series = data
    else:
        raise TypeError("data must be a pandas Series or a single-column DataFrame")

    if not isinstance(initial_train_size, int) or initial_train_size <= 0:
        raise ValueError("initial_train_size must be a positive integer")
    if not isinstance(test_size, int) or test_size <= 0:
        raise ValueError("test_size must be a positive integer")

    if len(series) < initial_train_size + test_size:
        raise ValueError("data must contain at least initial_train_size + test_size observations")

    train = series.iloc[:initial_train_size].copy()
    test = series.iloc[initial_train_size:initial_train_size + test_size].copy()

    forecasts = []
    for step in range(test_size):
        train_window = series.iloc[:initial_train_size + step].copy()
        actual_value = float(test.iloc[step])

        model = arch_model(train_window, mean="constant", vol="Garch", p=p, q=q, dist="normal")
        fit = model.fit(disp="off")

        try:
            forecast = fit.forecast(horizon=1)
            predicted_value = float(forecast.mean.iloc[-1, 0])
        except Exception:
            predicted_value = float(train_window.mean())

        forecasts.append({
            "actual": actual_value,
            "forecast": predicted_value,
            "step": step,
        })

    forecast_df = pd.DataFrame(forecasts)
    errors = forecast_df["actual"] - forecast_df["forecast"]

    metrics = {
        "mae": float(np.mean(np.abs(errors))),
        "mse": float(np.mean(errors**2)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }

    return {
        "train": train,
        "test": test,
        "forecast": forecast_df,
        "metrics": metrics,
    }, fit