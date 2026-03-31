# This module contains all the functions to run the stock simulator
import numpy as np
import pandas as pd

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

def gbm(n_steps = 252, T=1, n_paths = 5,mu = 0.1, sigma = 0.2,S0 = 100):
    '''Simulate Geometric Brownian Motion paths.
        Parameters:
         n_steps: Number of time steps
         T: Total time
         n_paths: Number of paths to simulate  
         mu: Drift coefficient
         sigma: Volatility coefficient
         S0: Initial stock price
        Returns:
        S : Simulated GBM paths as a numpy array of shape (n_steps, n_paths)
        times : Time points corresponding to the simulated paths as a numpy array of shape (n_steps,)
    '''
    dt = T/n_steps
    t = np.arange(0, T, dt).reshape(n_steps,1)
    W,times = wiener_process(n_steps=n_steps, n_paths=n_paths)
    S = np.full(shape = n_paths, fill_value =S0).reshape(1,n_paths) * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return S, times

def simple_moving_average(prices, window_size, path=None):
    '''
    Simple Moving Average (SMA) calculation for a given array of prices.
    Parameters:
    prices: A numpy array of stock prices.
    window_size: The number of periods to calculate the average over.
    path: The path index for which to calculate the SMA.
    Returns:
    sma: A pandas series of the SMA values.
    '''
    if prices.shape[1] > 1 and path is None:
        raise ValueError("Path index must be specified for multi-path data.")
    if path is not None and (path < 0 or path >= prices.shape[1]):
        raise ValueError("Path index out of bounds.")
    if path is not None:
        prices = prices[:, path]
    price_series = pd.Series(prices.flatten())
    sma = price_series.rolling(window=window_size,closed="both").mean().dropna()
    return sma

def exponential_moving_average(prices, window_size, path=None):
    '''
    Exponential Moving Average (EMA) calculation for a given array of prices.
    prices: A numpy array of stock prices.
    window_size: The number of periods to calculate the average over.
    path: The path index for the stock prices.
    return: A pandas series of EMA values
    '''
    if prices.shape[1] > 1 and path is None:
        raise ValueError("Path index must be specified for multi-path data.")
    if path is not None and (path < 0 or path >= prices.shape[1]):
        raise ValueError("Path index out of bounds.")
    if path is not None:
        prices = prices[:, path]
    ema = pd.Series(prices).ewm(span=window_size, adjust=False).mean()
    return ema

def rolling_volatility(prices, window_size, path=None, annualize=True):
    '''Calculate rolling volatility for a given price series.
    Parameters:
    prices: numpy array of price data
    window_size: int, size of the rolling window
    path: int, index of the path to analyze (for multi-path data)
    annualize: bool, whether to annualize the volatility
    Returns:
    volatility: pandas Series of rolling volatility values
    '''
    if prices.shape[1] > 1 and path is None:
        raise ValueError("Path index must be specified for multi-path data.")
    if path is not None and (path < 0 or path >= prices.shape[1]):
        raise ValueError("Path index out of bounds.")
    if path is not None:
        prices = prices[:, path]
    price_series = pd.Series(np.diff(np.log(prices)).flatten())
    volatility = price_series.rolling(window_size).std() * np.sqrt(252)
    if not annualize:
        volatility /= np.sqrt(252)
    return volatility