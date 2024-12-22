#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def cal_sample_covariance(X):
    """
    Parameter:
        X: the sample return data
    Return:
        the estimated covariance matrix
    """
    demean_X = X - X.mean(axis = 0)
    return demean_X.T @ demean_X

def negative_portfolio_Sharpe_Ratio(weight, cov, mean):
    """
    Parameter:
        weight: portfolio weights
        cov: the estimated covariance matrix
        mean: the estimated mean vector
    Return:
        Negative portfolio Sharpe Ratio
    """
    return -(weight.T @ mean) / np.sqrt(weight.T @ cov @ weight.T)
    
def portfolio_optimization(cov, mean):
    """
    Parameter:
        cov: the estimated covariance matrix
        mean: the estimated mean vector
    Return:
        Portfolio weights
    """
    num = cov.shape[0]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, *[{'type': 'ineq', 'fun': lambda w, i=i: w[i]} for i in range(num)])
    bounds = [(None, None) for _ in range(num)]
    result = minimize(negative_portfolio_Sharpe_Ratio, x0 = np.ones(num) / num, args=(cov, mean, ),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

#%%
if __name__ == "__main__":
    ETF_return = pd.read_csv("ETF_return_data.csv", index_col=0)
    ETF_return.dropna(inplace = True)
    ETF_return_insample = ETF_return.loc[ETF_return.loc[:, "bob"] <= '2024-09-02']

    #%%
    ETF_return_matrix = ETF_return_insample.loc[:, "return"].values.reshape((-1, 12), order="F")

    #%%
    ETF_return_expectation = ETF_return_matrix.mean(axis = 0)
    sample_cov = cal_sample_covariance(ETF_return_matrix)
    weight = portfolio_optimization(sample_cov, ETF_return_expectation)
    sample_cov_return = ETF_return_matrix @ weight
    equal_weight_return = ETF_return_matrix @ np.ones((12, 1)) * 1/12

    #%%
    sample_cov_return = np.array(sample_cov_return)
    equal_weight_return = np.array(equal_weight_return)
    print(f"MVE mean: {sample_cov_return.mean():.3e}")
    print(f"MVE std: {sample_cov_return.std():.3e}")
    print(f"MVE Sharpe Ratio: {sample_cov_return.mean()/sample_cov_return.std()*np.sqrt(252):.3e}")
    print(f"Equal-weighted mean: {equal_weight_return.mean():.3e}")
    print(f"Equal-weighted std: {equal_weight_return.std():.3e}")
    print(f"Equal-weighted std: {equal_weight_return.mean()/equal_weight_return.std()*np.sqrt(252):.3e}")

    #%%
    import matplotlib.dates as mdates

    dates = ETF_return_insample.loc[:, "bob"].unique()

    plt.plot(dates, np.exp(np.cumsum(sample_cov_return)), linewidth = 2, label = "MVE Portfolio")
    plt.plot(dates, np.exp(np.cumsum(equal_weight_return)), linewidth = 2, label = "Equal-weighted Portfolio")
    plt.ylabel("Assets Value")
    plt.xlabel("Dates")
    plt.legend() 
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=50)) 
    plt.gcf().autofmt_xdate()
    plt.plot()
    # %%
