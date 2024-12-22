#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from MVE_Portfolio_Construction import *

#%%
tau = 0.05
ETF_return = pd.read_csv("ETF_return_data.csv", index_col=0)
ETF_return.dropna(inplace = True)
# I use the mean return of previous 3 month to represent prior return
ETF_return_sample = ETF_return.loc[(ETF_return.loc[:, "bob"] <= '2024-09-02') | (ETF_return.loc[:, "bob"] > '2024-06-02')]
ETF_return_matrix_sample = ETF_return_sample.loc[:, "return"].values.reshape((-1, 12), order="F")
pi = ETF_return_matrix_sample.mean(axis = 0)
Sigma = sample_cov = cal_sample_covariance(ETF_return_matrix_sample)
ETF_return_outsample = ETF_return.loc[(ETF_return.loc[:, "bob"] > '2024-09-02')]
ETF_return_matrix_outsample = ETF_return_outsample.loc[:, "return"].values.reshape((-1, 12), order="F")

#%%
# Matrix P
P = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

# Vector q
q = np.array([
    0.015,
    0.02,
    -0.005,
    0.02,
    0.02,
    -0.01,
    0.01,
    0.01,
    0.01,
    -0.01
])

# Matrix Omega
Omega = np.diag([
    0.1**2, 
    0.15**2, 
    0.3**2, 
    0.2**2, 
    0.1**2, 
    0.1**2, 
    0.2**2, 
    0.2**2, 
    0.25**2, 
    0.2**2
])

# Posterior mean
middle_term = np.linalg.inv(tau * np.linalg.inv(Sigma) + P.T @ np.linalg.inv(Omega) @ P)
mu_BL = middle_term @ ((np.linalg.inv(tau * Sigma) @ pi) + (P.T @ np.linalg.inv(Omega) @ q))

# Posterior covariance
Sigma_BL = Sigma + middle_term

# BL, MVE and equal
weight = portfolio_optimization(Sigma_BL, mu_BL)
BL_return = ETF_return_matrix_outsample @ weight
weight = portfolio_optimization(Sigma, pi)
sample_cov_return = ETF_return_matrix_outsample @ weight
equal_weight_return = ETF_return_matrix_outsample @ np.ones((12, 1)) * 1/12

#%%
sample_cov_return = np.array(sample_cov_return)
equal_weight_return = np.array(equal_weight_return)
BL_return = np.array(BL_return)
print(f"MVE mean: {sample_cov_return.mean():.3e}")
print(f"MVE std: {sample_cov_return.std():.3e}")
print(f"MVE Sharpe Ratio: {sample_cov_return.mean()/sample_cov_return.std()*np.sqrt(252):.3e}")
print(f"Equal-weighted mean: {equal_weight_return.mean():.3e}")
print(f"Equal-weighted std: {equal_weight_return.std():.3e}")
print(f"Equal-weighted std: {equal_weight_return.mean()/equal_weight_return.std()*np.sqrt(252):.3e}")
print(f"BL mean: {BL_return.mean():.3e}")
print(f"BL std: {BL_return.std():.3e}")
print(f"BL Sharpe Ratio: {BL_return.mean()/BL_return.std()*np.sqrt(252):.3e}")

#%%
import matplotlib.dates as mdates

dates = ETF_return_outsample.loc[:, "bob"].unique()

plt.plot(dates, np.exp(np.cumsum(sample_cov_return)), linewidth = 2, label = "MVE Portfolio")
plt.plot(dates, np.exp(np.cumsum(equal_weight_return)), linewidth = 2, label = "Equal-weighted Portfolio")
plt.plot(dates, np.exp(np.cumsum(BL_return)), linewidth = 2, label = "Black Litterman Portfolio")
plt.axvline(x=16, color='red', linestyle='--', linewidth=1)
plt.ylabel("Assets Value")
plt.xlabel("Dates")
plt.legend() 
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10)) 
plt.gcf().autofmt_xdate()
plt.plot()