# Quantitative Finance Lab: Portfolio Optimization and Efficient Frontier

**Modern Portfolio Theory in Action**  
This notebook implements Harry Markowitz's Modern Portfolio Theory using real market data. We construct the Efficient Frontier and find the portfolio with the highest Sharpe Ratio.

**Assets Included**:
- **SPY** – S&P 500 ETF (broad US stocks)
- **GLD** – Gold ETF (commodity hedge)
- **BND** – Total Bond ETF (fixed income)
- **AAPL** – Apple Inc.
- **MSFT** – Microsoft Corp.

## Learning Objectives
- Download historical data with `yfinance`
- Compute daily returns, annualized expected returns, and covariance matrix
- Simulate random portfolios to visualize the Efficient Frontier
- Optimize weights to maximize the Sharpe Ratio (no short-selling)
- Visualize and interpret the results

## Required Libraries

```python
# Install yfinance (run only if needed)
!pip install yfinance --quiet

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Nice plotting style
plt.style.use('seaborn-v0_8-darkgrid')

# Define tickers
tickers = ['SPY', 'GLD', 'BND', 'AAPL', 'MSFT']

# Download ~7 years of adjusted closing prices
data = yf.download(tickers, start='2018-01-01', end='2025-12-21')['Adj Close']

# Show first few rows
data.head()

# Daily returns
returns = data.pct_change().dropna()

returns.head()

# Annualized mean returns (252 trading days)
mean_returns = returns.mean() * 252

# Annualized covariance matrix
cov_matrix = returns.cov() * 252

print("Annualized Mean Returns")
print(mean_returns.round(4))
print("\nAnnualized Covariance Matrix")
print(cov_matrix.round(4))

def portfolio_return(weights, mean_returns):
    return np.dot(weights, mean_returns)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    p_return = portfolio_return(weights, mean_returns)
    p_vol = portfolio_volatility(weights, cov_matrix)
    return -(p_return - risk_free_rate) / p_vol

num_portfolios = 15000
num_assets = len(tickers)

# Store results: [return, volatility, sharpe]
results = np.zeros((3, num_portfolios))

np.random.seed(42)  # Reproducible results

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= weights.sum()
    
    p_return = portfolio_return(weights, mean_returns)
    p_vol = portfolio_volatility(weights, cov_matrix)
    p_sharpe = p_return / p_vol
    
    results[0, i] = p_return
    results[1, i] = p_vol
    results[2, i] = p_sharpe

plt.figure(figsize=(12, 8))
scatter = plt.scatter(results[1,:], results[0,:], 
                      c=results[2,:], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Expected Return')
plt.title('Efficient Frontier - Simulated Portfolios')
plt.show()