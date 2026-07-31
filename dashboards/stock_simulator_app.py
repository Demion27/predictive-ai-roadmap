import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from stock_simulator import (
    calculate_moving_average,
    calculate_rolling_volatility,
    simulate_garch,
    simulate_gbm,
)


@st.cache_data(show_spinner=False)
def run_gaussian_simulation(S0, mu, sigma, T, n_steps, n_paths, seed):
    return simulate_gbm(
        S0=float(S0),
        mu=float(mu),
        sigma=float(sigma),
        T=float(T),
        n_steps=int(n_steps),
        n_paths=int(n_paths),
        seed=int(seed),
    )


@st.cache_data(show_spinner=False)
def run_garch_simulation(n_steps, garch_mu, garch_omega, garch_alpha, garch_beta, garch_start_price, n_paths, seed, garch_burn):
    return simulate_garch(
        n_steps=int(n_steps),
        mu=float(garch_mu),
        omega=float(garch_omega),
        alpha=float(garch_alpha),
        beta=float(garch_beta),
        start_price=float(garch_start_price),
        n_paths=int(n_paths),
        seed=int(seed),
        burn=int(garch_burn),
    )


st.set_page_config(page_title="Stock Simulation Dashboard", layout="wide")
st.title("Gaussian vs. GARCH Simulation Dashboard")

st.sidebar.header("Simulation controls")
selected_modes = st.sidebar.multiselect(
    "Choose one or more simulations",
    ["Gaussian", "GARCH", "Both"],
    default=["Gaussian", "GARCH"],
)

# normalize selection for display
if "Both" in selected_modes:
    selected_modes = ["Gaussian", "GARCH"]

st.sidebar.markdown("### Gaussian parameters")
st.sidebar.caption("Hint: drift is typically between -0.5 and 0.5, and sigma is usually between 0.01 and 0.5.")
S0 = st.sidebar.number_input("Initial price (S0)", value=100.0, min_value=1.0, max_value=10000.0, step=1.0)
mu = st.sidebar.number_input("Drift (mu)", value=0.08, min_value=-1.0, max_value=1.0, step=0.0001, format="%.4f")
sigma = st.sidebar.number_input("Volatility (sigma)", value=0.25, min_value=0.001, max_value=2.0, step=0.0001, format="%.4f")
T = st.sidebar.number_input("Time horizon (T)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
n_steps = st.sidebar.number_input("Number of steps", value=252, min_value=20, max_value=1000, step=6)
n_paths = st.sidebar.number_input("Number of paths", value=1, min_value=1, max_value=10, step=1)
seed = st.sidebar.number_input("Random seed", value=42, min_value=0, max_value=100000, step=1)

st.sidebar.markdown("### GARCH parameters")
st.sidebar.caption("Hint: omega should be positive, and alpha + beta should stay below 1 for a stable process.")
garch_mu = st.sidebar.number_input("GARCH mean return (mu)", value=0.0003, min_value=-0.5, max_value=0.5, step=0.0001, format="%.4f")
garch_omega = st.sidebar.number_input("GARCH omega", value=0.00001, min_value=0.0, max_value=1.0, step=0.000001, format="%.6f")
garch_alpha = st.sidebar.number_input("GARCH alpha", value=0.1, min_value=0.0, max_value=0.99, step=0.0001, format="%.4f")
garch_beta = st.sidebar.number_input("GARCH beta", value=0.87, min_value=0.0, max_value=0.99, step=0.0001, format="%.4f")
garch_start_price = st.sidebar.number_input("GARCH start price", value=100.0, min_value=1.0, max_value=10000.0, step=1.0)
garch_burn = st.sidebar.number_input("GARCH burn-in", value=100, min_value=0, max_value=1000, step=1)

window_size = st.sidebar.slider("Moving-average and volatility window", 2, 60, 20)
selected_ma_method = st.sidebar.selectbox("Moving-average method", ["Simple", "Exponential"], index=0)
selected_vol_method = st.sidebar.selectbox("Rolling-volatility method", ["Log returns", "Percent change"], index=0)

ma_method = "simple" if selected_ma_method == "Simple" else "exponential"
vol_method = "log" if selected_vol_method == "Log returns" else "pct"

if "Gaussian" in selected_modes:
    gaussian_df = run_gaussian_simulation(
        S0=float(S0),
        mu=float(mu),
        sigma=float(sigma),
        T=float(T),
        n_steps=int(n_steps),
        n_paths=int(n_paths),
        seed=int(seed),
    )
    gaussian_ma = calculate_moving_average(
        gaussian_df,
        window_size=int(window_size),
        method=ma_method,
        columns=list(gaussian_df.columns),
    )
    gaussian_vol = calculate_rolling_volatility(
        gaussian_df,
        window_size=int(window_size),
        columns=list(gaussian_df.columns),
        method=vol_method,
        annualize=False,
    )

if "GARCH" in selected_modes:
    try:
        prices_garch, true_vol, returns = run_garch_simulation(
            n_steps=int(n_steps),
            garch_mu=float(garch_mu),
            garch_omega=float(garch_omega),
            garch_alpha=float(garch_alpha),
            garch_beta=float(garch_beta),
            garch_start_price=float(garch_start_price),
            n_paths=int(n_paths),
            seed=int(seed),
            garch_burn=int(garch_burn),
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    garch_ma = calculate_moving_average(
        prices_garch,
        window_size=int(window_size),
        method=ma_method,
        columns=list(prices_garch.columns),
    )
    garch_vol = calculate_rolling_volatility(
        prices_garch,
        window_size=int(window_size),
        columns=list(prices_garch.columns),
        method=vol_method,
        annualize=False,
    )

st.subheader("Simulation overview")

if "Gaussian" in selected_modes:
    st.markdown("#### Gaussian simulation")
    st.dataframe(gaussian_df.head())

if "GARCH" in selected_modes:
    st.markdown("#### GARCH simulation")
    st.dataframe(prices_garch.head())

if "Gaussian" in selected_modes:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for col in gaussian_df.columns:
        axes[0].plot(gaussian_df[col], label=f"{col}")
        axes[0].plot(gaussian_ma[col], linestyle="--", alpha=0.7, linewidth=1.0)
    axes[0].set_title("Gaussian price paths and moving averages")
    axes[0].legend(ncol=2, fontsize=8)
    for col in gaussian_df.columns:
        axes[1].plot(gaussian_vol[f"{col} volatiltity"], label=f"{col}")
    axes[1].set_title("Gaussian rolling volatility")
    axes[1].set_xlabel("Step")
    plt.tight_layout()
    st.pyplot(fig)

if "GARCH" in selected_modes:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for col in prices_garch.columns:
        axes[0].plot(prices_garch[col], label=f"{col}")
        axes[0].plot(garch_ma[col], linestyle="--", alpha=0.7, linewidth=1.0)
    axes[0].set_title("GARCH price paths and moving averages")
    axes[0].legend(ncol=2, fontsize=8)
    for col in prices_garch.columns:
        axes[1].plot(garch_vol[f"{col} volatiltity"], label=f"{col}")
    axes[1].set_title("GARCH rolling volatility")
    axes[1].set_xlabel("Step")
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("Comparison summary")
comparison = []
if "Gaussian" in selected_modes:
    comparison.append({
        "model": "Gaussian",
        "final_price": float(gaussian_df.iloc[-1].mean()),
        "mean_price": float(gaussian_df.mean().mean()),
        "volatility_mean": float(gaussian_vol.mean().mean()),
    })
if "GARCH" in selected_modes:
    comparison.append({
        "model": "GARCH",
        "final_price": float(prices_garch.iloc[-1].mean()),
        "mean_price": float(prices_garch.mean().mean()),
        "volatility_mean": float(garch_vol.mean().mean()),
    })

st.dataframe(pd.DataFrame(comparison))
