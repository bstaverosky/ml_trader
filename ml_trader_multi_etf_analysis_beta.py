#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:57:04 2021

@author: bstaverosky
"""
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
#import pandas_datareader.data as web
import seaborn as sns
import sklearn
import empyrical as ep
import statsmodels.api as sm
from sklearn.ensemble import (BaggingRegressor, RandomForestRegressor, AdaBoostRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from os import path
import pyfolio as pf
import yfinance as yf
import talib as talib
import time
exec(open("/home/brian/Documents/projects/ml_trader/ml_trader_functions.py").read())

def percentage_change(col1,col2):
    return ((col2 - col1) / col1) * 100

def make_perf_output(ticker,strat_series,bmk_series,lw,pw,nest,mdepth,sframe,min_length,algo):
    # Performance Data Frame
    import pandas as pd
    from datetime import datetime
    import empyrical as ep
    from sklearn.metrics import mean_squared_error as MSE
    from sklearn.metrics import accuracy_score
    if min_length == False:
        perf = pd.DataFrame({
            'Date Run': datetime.today().strftime('%Y-%m-%d'),
            'Ticker': ticker,
            'Model': algo,
            'Prediction Window': [0],
            'Lookback Window': [0],
            'Number of Estimators': [0],
            'Max Depth': [0],
            'Strategy Annual Return': [0],
            'Benchmark Annual Return': [0],
            'Active Annualized Return': [0],
            'Cumulative Returns': [0],
            'Sharpe Ratio': [0],
            'Sortino Ratio': [0],
            'Max Drawdown': [0],
            'Mean Squared Error': [0],
            'Baseline': [0],
            'Accuracy': [0],
            'Skill': [0]
        })
        return(perf)
                
    elif min_length == True:
        perf = pd.DataFrame({
            'Date Run': datetime.today().strftime('%Y-%m-%d'),
            'Ticker': ticker,
            'Model': algo,
            'Prediction Window': [pw],
            'Lookback Window': [lw],
            'Number of Estimators': nest,
            'Max Depth': mdepth,
            'Strategy Annual Return': ep.cagr(strat_series),
            'Benchmark Annual Return': ep.cagr(bmk_series),
            'Active Annualized Return': ep.cagr(strat_series)-ep.cagr(bmk_series),
            'Cumulative Returns': ep.cum_returns_final(strat_series)*100,
            'Sharpe Ratio': ep.sharpe_ratio(strat_series),
            'Sortino Ratio': ep.sortino_ratio(strat_series),
            'Max Drawdown': ep.max_drawdown(strat_series),
            'Mean Squared Error': MSE(sframe.loc[:,"pwret"], sframe.loc[:,"signal"])**(1/2),
            'Baseline': (sframe.loc[:,"pwret_bin"] == 1).sum()/len(sframe),
            'Accuracy': accuracy_score(sframe.loc[:,"pwret_bin"], sframe.loc[:,"signal_bin"]),
            'Skill': accuracy_score(sframe.loc[:,"pwret_bin"], sframe.loc[:,"signal_bin"])-(sframe.loc[:,"pwret_bin"] == 1).sum()/len(sframe)
            })
        return (perf)

def do_etfuniv_backtest(tickers, mdepth=3, nest=2, pw=63, lw=252, algo="xg"):
    for x in tickers:
        print(x)
        asset = yf.download(x, start='1900-01-01', progress=True)
        
        if len(asset.index) < 2560:    
            perf = make_perf_output(ticker = x, strat_series=0, bmk_series =0, lw=lw, pw = pw, nest = nest, mdepth = mdepth, sframe = 0, min_length=False, algo = algo)
    
            # Save to CSV
            if path.exists('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv") == True:
                perf.to_csv('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv", mode = 'a', header = False)
            elif path.exists('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv") == False:
                perf.to_csv('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv", header = True)
                
        elif len(asset.index) >= 2560:
        
            # Calculate signals
            # SMA RATIO
            asset['sma_rat'] = np.log(talib.SMA(asset['Close'], timeperiod=21)/talib.SMA(asset['Close'], timeperiod = 252))
            
            # VOLUME RATIO
            asset["vme_rat"] = np.log(talib.SMA(asset['Volume'], timeperiod=10)/talib.SMA(asset['Volume'], timeperiod = 252))
                
            # VOLATILITY RATIO
            for i in range(len(asset.index)):
                asset.loc[asset.index[i], "stvol"] = np.std(np.diff(np.log(asset.loc[asset.index[1:i], "Close"].tail(65))))
                asset.loc[asset.index[i], "ltvol"] = np.std(np.diff(np.log(asset.loc[asset.index[1:i], "Close"].tail(252))))
                asset.loc[asset.index[i], "vol_rat"] = asset.loc[asset.index[i], "stvol"]/asset.loc[asset.index[i], "ltvol"]
                
            # PRICE TO HIGH
            for i in range(len(asset.index)):
                asset.loc[asset.index[i], "p2h"] = asset.loc[asset.index[i], "Close"]/np.max(asset.loc[asset.index[(i-252):(i-1)], "Close"])
                
            # Get Daily Return
            asset['dayret'] = asset['Close'].pct_change()
            
            # Get Prediction Window Return
            asset['closelag5'] = asset['Close'].shift(pw)
            asset['pwret'] = percentage_change(asset['closelag5'],asset['Close'])
            asset['pwret'] = asset['pwret'].shift(-pw-1)
            
            # CLEAN DATAFRAME
            df = asset[['sma_rat', 'vme_rat', 'vol_rat', 'p2h', 'pwret']]
            df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
            df = df.dropna()
            print("still working")
            
            if len(df.index) < ((lw+pw)*2):
            
                print("ticker does not have enough history")    
                # Performance Data Frame
                perf = make_perf_output(ticker = x, strat_series=0, bmk_series = 0, lw=lw, pw = pw, nest = nest, mdepth = mdepth, sframe = 0, min_length=False)
        
                # Save to CSV
                if path.exists('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv") == True:
                    perf.to_csv('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv", mode = 'a', header = False)
                elif path.exists('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv") == False:
                    perf.to_csv('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv", header = True)
                
            elif len(df.index) >= ((lw+pw)*2) :
                #j = lw
                predf = pd.DataFrame(columns = ["pred"])
                
                if algo == "xg":
                # Create rolling window trainset
                    for i in range((lw+pw), len(df.index)-1):
                        gb = GradientBoostingRegressor(n_estimators = nest,
                                                       max_depth=mdepth,
                                                       random_state=2)
                        # Make trainsets
                        xtrain = df.loc[df.index[i-(lw+pw)]:df.index[i-pw],['sma_rat', 'vol_rat', 'p2h']]
                        ytrain = df.loc[df.index[i-(lw+pw)]:df.index[i-pw],['pwret']]
                            
                        # Make testsets
                        xtest = df.loc[[df.index[i+1]],['sma_rat', 'vol_rat', 'p2h']]    
                        ytest = df.loc[[df.index[i+1]],['pwret']]
                        
                        gb.fit(xtrain, ytrain)
                        y_pred = gb.predict(xtest)
                   
                        lframe = pd.DataFrame(y_pred, columns = ["pred"], index = ytest.index)
                        predf = predf.append(lframe)
                
                elif algo == "linreg":
                # Create rolling window trainset
                    for i in range((lw+pw), len(df.index)-1):
                        
                        # Make trainsets
                        xtrain = df.loc[df.index[i-(lw+pw)]:df.index[i-pw],['sma_rat', 'vol_rat', 'p2h']]
                        ytrain = df.loc[df.index[i-(lw+pw)]:df.index[i-pw],['pwret']]
                            
                        # Make testsets
                        xtest = df.loc[[df.index[i+1]],['sma_rat', 'vol_rat', 'p2h']]    
                        ytest = df.loc[[df.index[i+1]],['pwret']]
                        
                        model = sm.OLS(ytrain, xtrain).fit()
                        y_pred = model.predict(xtest)
                        
                        lframe = pd.DataFrame(y_pred, columns = ["pred"], index = ytest.index)
                        predf = predf.append(lframe)
                        
                        #model.summary()
                    
                # Put predictions back on original data frame
                # And convert y_pred so it can be added to dataframe
                sframe = df
                sframe['signal'] = predf
                sframe['signal'] = sframe['signal'].shift(1)
                sframe['return'] = asset['dayret']
                
                if len(sframe) < 5:
                    print("no strategy history")
                    
                elif len(sframe) > 5:
                
                    # Create the strategy return performance
                    for i in range(len(sframe.index)):
                        if sframe.loc[sframe.index[i], "signal"] > 0:
                            sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*1
                        else:
                            sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*0
                            
                    bmk_series = sframe.loc[:,"return"]
                    strat_series = sframe.loc[:,"strat"]
                    sframe = sframe.dropna()
                    #pf.create_simple_tear_sheet(returns = strat_series, benchmark_rets=bmk_series)
                    # Convert regression prediction to categories to binaries
    
                    for i in range(len(sframe.index)):
                        if sframe.loc[sframe.index[i], "pwret"] >= 0:
                            sframe.loc[sframe.index[i], "pwret_bin"] = 1
                        else:
                            sframe.loc[sframe.index[i], "pwret_bin"] = 0
                    
                    for i in range(len(sframe.index)):
                        if sframe.loc[sframe.index[i], "signal"] >= 0:
                            sframe.loc[sframe.index[i], "signal_bin"] = 1
                        else:
                            sframe.loc[sframe.index[i], "signal_bin"] = 0
                                    
                    print(x)
                    # Performance Data Frame                
                    perf = make_perf_output(ticker=x, strat_series=strat_series, bmk_series=bmk_series, lw=lw, pw=pw, nest=nest, mdepth=mdepth, sframe=sframe, min_length=True, algo = algo)
                    # Save to CSV
                    if path.exists('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv") == True:
                        perf.to_csv('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv", mode = 'a', header = False)
                    elif path.exists('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv") == False:
                        perf.to_csv('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv", header = True)

def check_conditions(subset):
    if (all(val in subset['Model'].values for val in ["xg", "linreg"]) and all(val in subset['Prediction Window'].values for val in [5,10,21,42,63,126]) and all(val in subset['Lookback Window'].values for val in [252, 504, 756, 1260, 2520])):
        return True
    return False
##### PARAMETERS #####

# Archive current database?
dbdel = False 

# Current Algos
algos = ["xg", "linreg"]

# max depth
mdepth = 3

# Number of estimators (trees)
nest = 2

#prediction window
pw = 63

#lookback window
lw = 252

# Load asset list

tickers = pd.read_csv("/home/brian/Documents/projects/ml_trader/ETF_Database.csv", header=None)
tickers = tickers.iloc[:,0].tolist()

# Load tickers already run

if path.exists('/home/brian/Documents/projects/ml_trader' + "/" + "ETF_universe_results" + "_ml_trader.csv") == True:
    rframe = pd.read_csv("/home/brian/Documents/projects/ml_trader/ETF_universe_results_ml_trader.csv", on_bad_lines='skip')
    rticks = rframe.loc[:,"Ticker"].tolist()
    
    verified_tickers = []
    # loop through tickers
    for ticker in rticks:
        # Subset dataframe for specific ticker
        subset = rframe[rframe['Ticker']==ticker]
        # check to see if all models, prediction windows and lookback windows have been tested
        if check_conditions(subset):
            # add ticker to verified tickers list
            verified_tickers.append(ticker)
            
    tickers = list(set(tickers) - set(verified_tickers))

### DO THE WORK ###

for x, y, z in product(["xg", "linreg"], [5,10,21,42,63,126], [252, 504, 756, 1260, 2520]):
    do_etfuniv_backtest(tickers, mdepth=3, nest=2, pw=y, lw=z, algo = x)   
            





