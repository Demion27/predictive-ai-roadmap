import numpy as np
import pandas as pd
import pytest

from src.stock_simulator import (
    calculate_moving_average,
    calculate_rolling_volatility,
    simulate_garch,
    simulate_gbm,
    train_garch_rolling_forecast,
    wiener_process,
)


def test_wiener_process_returns_expected_shapes():
    w, times = wiener_process(n_steps=10, T=1.0, n_paths=3)

    assert isinstance(w, np.ndarray)
    assert isinstance(times, np.ndarray)
    assert w.shape == (10, 3)
    assert times.shape == (10,)
    assert np.allclose(times, np.linspace(0, 1.0, 10))


def test_simulate_gbm_returns_dataframe_with_expected_index_and_columns():
    df = simulate_gbm(S0=100.0, mu=0.05, sigma=0.2, T=1.0, n_steps=12, n_paths=2, seed=7)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["path_0", "path_1"]
    assert df.index.name == "time"
    assert df.shape == (13, 2)
    assert np.isclose(df.iloc[0, 0], 100.0)
    assert np.isclose(df.iloc[0, 1], 100.0)


def test_calculate_moving_average_supports_series_and_dataframe_inputs():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    result_series = calculate_moving_average(series, window_size=2)

    assert isinstance(result_series, pd.Series)
    assert result_series.isna().sum() == 1

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    result_df = calculate_moving_average(df, window_size=2, columns=["a", "b"])

    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == ["a", "b"]
    assert result_df.shape[0] == 3

    with pytest.raises(ValueError):
        calculate_moving_average(series, window_size=0)

    with pytest.raises(KeyError):
        calculate_moving_average(df, columns=["missing"])


def test_calculate_rolling_volatility_returns_expected_structure():
    df = pd.DataFrame({"price": [100.0, 101.0, 99.0, 102.0, 100.0]})

    result = calculate_rolling_volatility(df, window_size=2, columns="price", annualize=False)
    pct_result = calculate_rolling_volatility(df, window_size=2, columns="price", method="pct", annualize=False)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["price volatiltity"]
    assert result.shape[0] == 3
    assert isinstance(pct_result, pd.DataFrame)
    assert pct_result.shape[0] == 3

    with pytest.raises(TypeError):
        calculate_rolling_volatility([1, 2, 3], window_size=2)

    with pytest.raises(KeyError):
        calculate_rolling_volatility(df, columns=["missing"])


def test_simulate_garch_returns_prices_volatility_and_returns():
    prices, true_vol, returns = simulate_garch(
        n_steps=10,
        mu=0.001,
        omega=0.00001,
        alpha=0.1,
        beta=0.8,
        start_price=100.0,
        n_paths=1,
        seed=3,
        burn=10,
    )

    assert isinstance(prices, pd.DataFrame)
    assert isinstance(true_vol, pd.DataFrame)
    assert isinstance(returns, pd.DataFrame)
    assert prices.shape[0] == 11
    assert true_vol.shape[0] == 11
    assert returns.shape[0] == 10
    assert np.isclose(prices.iloc[0, 0], 100.0)

    with pytest.raises(ValueError):
        simulate_garch(n_steps=10, alpha=0.8, beta=0.3)


def test_train_garch_rolling_forecast_returns_expected_structure():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, 140))

    result, model = train_garch_rolling_forecast(
        returns,
        initial_train_size=100,
        test_size=20,
        p=1,
        q=1,
    )

    assert set(result.keys()) >= {"train", "test", "forecast", "metrics"}
    assert len(result["train"]) == 100
    assert len(result["test"]) == 20
    assert len(result["forecast"]) == 20
    assert {"actual", "forecast"}.issubset(result["forecast"].columns)
    assert model is not None

    for metric in ["mae", "mse", "rmse"]:
        assert metric in result["metrics"]

    with pytest.raises(ValueError):
        train_garch_rolling_forecast(returns.iloc[:10], initial_train_size=5, test_size=10)
