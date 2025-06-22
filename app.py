import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes Delta
def bs_delta(S, K, T, r, sigma, q):
    """
    Computes the Black-Scholes delta for a call option.
    """
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q*T) * norm.cdf(d1)

# Black-Scholes Call Price
def bs_call_price(S, K, T, r, sigma, q):
    """
    Computes the Black-Scholes price of a call option.
    """
    if T <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Simulate GBM paths
def simulate_gbm_paths(S0, T, r, sigma, q, n_steps, n_paths):
    """
    Simulates geometric Brownian motion paths.
    """
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        z = np.random.normal(0, 1, n_paths)
        paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * z)
    return paths

def main():
    with st.sidebar:
        n_rebalances = st.slider("Number of Rebalances", min_value=10, max_value=1000, value=20, step=10)
        n_paths = st.slider("Number of Simulations", min_value=10, max_value=500, value=50, step=10)
        S0 = st.slider("Initial Stock Price (S₀)", min_value=10, max_value=200, value=100, step=1)
        strike = st.slider("Strike Price (K)", min_value=10, max_value=200, value=100, step=1)
        T = st.slider("Time to Maturity (T, in years)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        r = st.slider("Risk-Free Rate (r, in %)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100.0
        sigma = st.slider("Volatility (σ, in %)", min_value=5.0, max_value=100.0, value=20.0, step=1.0) / 100.0
        q = st.slider("Dividend Yield (q, in %)", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100.0
  
    
    # Simulate stock price paths
    paths = simulate_gbm_paths(S0, T, r, sigma, q, n_rebalances, n_paths)
    dt = T / (n_rebalances + 1)

    # Initial Black-Scholes call price
    call_price = bs_call_price(S0, strike, T, r, sigma, q)
    
    # Simulate hedging strategy
    portfolio_values = []
    for i in range(n_paths):
        stock_prices = paths[i]
        cash = call_price - bs_delta(S0, strike, T, r, sigma, q) * S0  # Initial cash
        delta_prev = bs_delta(S0, strike, T, r, sigma, q)  # Initial delta
        for t in range(1, n_rebalances+1):
            T_t = T - t * dt
            delta = bs_delta(stock_prices[t], strike, T_t, r, sigma, q)
            # Update cash and delta
            cash = (
                cash * (1 + r * dt) 
                + delta_prev * stock_prices[t-1] * q * dt 
                - (delta - delta_prev) * stock_prices[t]
            )
            delta_prev = delta
        # Final portfolio value
        intrinsic_value = max(stock_prices[-1] - strike, 0)
        portfolio_value = cash + delta_prev * stock_prices[-1] - intrinsic_value
        portfolio_values.append(portfolio_value)

    # Plot the distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(portfolio_values, bins=50, color="blue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Portfolio Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)

    st.pyplot(fig)

if __name__ == "__main__":
    main()
