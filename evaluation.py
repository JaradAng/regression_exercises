import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import math

from pydataset import data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def plot_residuals(y, yhat):
    residual = y - yhat
    baseline_residual = y - y.mean()


    plt.figure(figsize = (11,5))

    plt.subplot(121)
    plt.scatter(y, baseline_residual)
    plt.axhline(y = 0, ls = ':')
    plt.xlabel('y')
    plt.ylabel('Residual')
    plt.title('Baseline Residuals')

    plt.subplot(122)
    plt.scatter(y, residual)
    plt.axhline(y = 0, ls = ':')
    plt.xlabel('y')
    plt.ylabel('Residual')
    plt.title('OLS model residuals')







def regression_errors(y, yhat):

    #Sum of Squared Errors
    SSE = ((y-yhat)** 2).sum()

    #Explained Sum of Squares
    ESS = ((yhat -y.mean())** 2).sum()
    #Total sum of sqaures
    TSS = ((y-y.mean())**2).sum
    #Mean square errors
    MSE = mean_square_error(y, yhat)
    # Root mean square errors
    RMSE = sqrt(MSE)

    return print(f'SSE = {SSE}, ESS = {ESS}, TSS = {TSS}, TSS = {TSS}, MSE = {MSE}, RMSE = {RMSE}')

def baseline_mean_errors(y):
    baseline_residuals = y - y.mean()
    sse_baseline = (baseline_residuals ** 2).sum()
    length = len(y)
    mse_baseline = sse_baseline / length
    rmse_basline = sqrt(mse_baseline)

    return print(f'Baseline Residuals = {baseline_residuals}, sse_baseline = {sse_baseline}, mse baseline = {mse_basline}, rmse baseline = {rmse_baseline}')




def better_than_baseline(y, yhat):
    #Sum of Squared Errors
    SSE = ((y-yhat)** 2).sum()

    #Baseline residuals
    baseline_residuals = y - y.mean()
    sse_baseline = (baseline_residuals ** 2).sum()
    
    if SSE < SSE_baseline:
        print ('The model preforms better than baseline')
    else:
        print ('Model underperforms')

