import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end, auto_adjust=True)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

# calculate weights for portfolio allocation
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# Monte Carlo Simulation
num_simulations = 10
num_days = 252

mean_matrix = np.full(shape=(num_days, len(weights)), fill_value=meanReturns)
mean_matrix = mean_matrix.T

portfolio_simulations = np.full(shape=(num_days, num_simulations), fill_value=0.0)
print(portfolio_simulations)

initial_portfolio_value = 10000

for i in range(num_simulations):
    Z = np.random.normal(size=(num_days, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    daily_returns = mean_matrix + np.inner(L, Z)
    portfolio_simulations[:, i] = np.cumprod(np.inner(weights, daily_returns.T) + 1) * initial_portfolio_value
    
plt.plot(portfolio_simulations)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of Portfolio Value Over One Year')
plt.show()