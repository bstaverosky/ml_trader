#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:57:04 2021

@author: bstaverosky
"""

### THIS IS A TEST FOR UPLOADING CHANGES FROM MY LAPTOP
### THIS IS A TEST EDIT FROM REMOTE
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import pandas_datareader.data as web
import seaborn as sns
import sklearn
import empyrical as ep
from sklearn.ensemble import (BaggingRegressor, RandomForestRegressor, AdaBoostRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
import pyfolio as pf
import yfinance as yf
import talib as talib
import os.path
from os import path
from datetime import datetime

##### PARAMETERS #####

algos = ["xg", "linreg"]

numest = [5,10,50,100,200]
#nest = 10
maxdepth = [1,2,3]
#mdepth = 2
#prediction window
predwindow = [3,5,10,21,63,126,256]
#pw = 10
#lookback window
lw = 256
# ticker
ticker = "SPY"
alg = "xg"

pw = 10
nest = 5
mdepth = 3


for pw in predwindow:
    for nest in numest:
        for mdepth in maxdepth:
            asset = yf.download(ticker, start='1900-01-01', progress=True)
            
            # Calculate signals
            # SMA RATIO
            asset['sma_rat'] = np.log(talib.SMA(asset['Close'], timeperiod=21)/talib.SMA(asset['Close'], timeperiod = 252))
            
            # VOL RATIO
            for i in range(len(asset.index)):
                asset.loc[asset.index[i], "stvol"] = np.std(np.diff(np.log(asset.loc[asset.index[1:i], "Close"].tail(65))))
                asset.loc[asset.index[i], "ltvol"] = np.std(np.diff(np.log(asset.loc[asset.index[1:i], "Close"].tail(252))))
                asset.loc[asset.index[i], "vol_rat"] = asset.loc[asset.index[i], "stvol"]/asset.loc[asset.index[i], "ltvol"]
                
            # PRICE TO HIGH
            for i in range(len(asset.index)):
                asset.loc[asset.index[i], "p2h"] = asset.loc[asset.index[i], "Close"]/np.max(asset.loc[asset.index[(i-252):(i-1)], "Close"])
                
            # Get Daily Return
            asset['dayret'] = asset['Close'].pct_change()
            
            # Get Weekly Return
            #test = asset.resample('W').ffill()
            asset['closelag'] = asset['Close'].shift(pw)
            #def percentage_change(col1,col2):
            #   return ((col2 - col1) / col1) * 100
            
            asset['weekret'] = percentage_change(asset['closelag'],asset['Close'])
            asset['weekret'] = asset['weekret'].shift(-(pw+1))
            
            # CLEAN DATAFRAME
            
            df = asset[['sma_rat', 'vol_rat', 'p2h', 'weekret']].tail(len(asset)-lw)
            #df = df.dropna()
            
            
            predf = pd.DataFrame(columns = ["pred"])
            
            # Create rolling window trainset
            for i in range((lw+pw), len(df.index)-1):
                print(i)
                
                if alg == "xg":
                    model = GradientBoostingRegressor(n_estimators = nest,
                                                      max_depth=mdepth,
                                                      random_state=2)
                elif alg == "linreg":
                    print("linreg")
                    model = LinearRegression()
                
                # Make trainsets
                xtrain = df.loc[df.index[i-(lw+pw)]:df.index[i-pw],['sma_rat', 'vol_rat', 'p2h']]
                ytrain = df.loc[df.index[i-(lw+pw)]:df.index[i-pw],['weekret']]
                    
                # Make testsets
                xtest = df.loc[[df.index[i+1]],['sma_rat', 'vol_rat', 'p2h']]    
                ytest = df.loc[[df.index[i+1]],['weekret']]
                type(ytest)
                    
                model.fit(xtrain, ytrain)
                y_pred = model.predict(xtest)
                    
                #results = pd.DataFrame(data = )
                    
                lframe = pd.DataFrame(y_pred, columns = ["pred"], index = ytest.index)
                predf = predf.append(lframe)
                
            # Put predictions back on original data frame
            # And convert y_pred so it can be added to dataframe
            # Put predictions back on original data frame
            # And convert y_pred so it can be added to dataframe
            
            sframe = df
            sframe['signal'] = predf
            sframe['signal'] = sframe['signal'].shift(1)
            sframe['return'] = asset['dayret']
            
            # CODE for continuous leverage option
            scaler = preprocessing.MinMaxScaler(feature_range=(0.75, 1.25))
            #scaler.fit_transform(sframe[["signal"]])
            
            # Fill in scaled signal
            for i in range(1281, len(sframe.index)):
                #sframe.iloc[(i-1280):i,[sframe.columns.get_loc("signal")]] = scaler.fit_transform(sframe.iloc[(i-1280):i,[sframe.columns.get_loc("signal")]])[1279:]
                sframe.iloc[i,[sframe.columns.get_loc("signal")]] = float(scaler.fit_transform(sframe.iloc[(i-1280):i,[sframe.columns.get_loc("signal")]])[1279:])
                #sframe.iloc[0:i,[sframe.columns.get_loc("signal")]] = scaler.fit_transform(sframe.iloc[0:i,[sframe.columns.get_loc("signal")]])
            
            
            # Create the strategy return performance
            # for i in range(len(sframe.index)):
            #     if sframe.loc[sframe.index[i], "signal"] > 0:
            #         sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*1.25
            #     else:
            #         sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*0.75
                    
                    
            #sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*sframe.loc[sframe.index[i], "signal"]
            sframe.loc[:,"strat"] = sframe.loc[:,"return"] * sframe.loc[:,"signal"]
            bmk_series = sframe.loc[:,"return"].tail(len(sframe)-(lw+pw))
            strat_series = sframe.loc[:,"strat"].tail(len(sframe)-(lw+pw))
            
            #tsheet = pf.create_simple_tear_sheet(returns = strat_series, benchmark_rets=bmk_series)
            
            pf.create_simple_tear_sheet(returns = strat_series, benchmark_rets=bmk_series)
            
            # Real-time forward week prediction
            rtpred = asset.loc[[asset.index[len(asset)-1]], ['sma_rat', 'vol_rat', 'p2h']]
            model.predict(rtpred)
            sframe = sframe.dropna()
            # Evaluate the test set RMSE
            #MSE(sframe.loc[:,"weekret"], sframe.loc[:,"signal"])**(1/2)
            
            
            # Performance Data Frame
            perf = pd.DataFrame({
                'Date Run': datetime.today().strftime('%Y-%m-%d'),
                'Ticker': ticker,
                'algo': alg,
                'Prediction Window': [pw],
                'Lookback Window': [lw],
                'Number of Estimators': nest,
                'Max Depth': mdepth,
                'Annualized Return': ep.cagr(strat_series),
                'Benchmark Annualized Return': ep.cagr(bmk_series),
                'Active Annualized Return': ep.cagr(strat_series)-ep.cagr(bmk_series),
                'Cumulative Returns': ep.cum_returns_final(strat_series)*100,
                'Sharpe Ratio': ep.sharpe_ratio(strat_series),
                'Benchmark Sharpe Ratio': ep.sharpe_ratio(bmk_series),
                'Sortino Ratio': ep.sortino_ratio(strat_series),
                'Max Drawdown': ep.max_drawdown(strat_series),
                'Mean Squared Error': MSE(sframe.loc[:,"weekret"], sframe.loc[:,"signal"])**(1/2)
                #'Baseline': (sframe.loc[:,"weekret_bin"] == 1).sum()/len(sframe),
                #'Accuracy': accuracy_score(sframe.loc[:,"weekret_bin"], sframe.loc[:,"signal_bin"]),
                #'Skill': accuracy_score(sframe.loc[:,"weekret_bin"], sframe.loc[:,"signal_bin"])-(sframe.loc[:,"weekret_bin"] == 1).sum()/len(sframe)
            })

            # Save to CSV
            if path.exists('/home/brian/Documents/projects/shared_projects/Data' + "/" + "adhoc" + "_model_rolling_strat.csv") == True:
                perf.to_csv('/home/brian/Documents/projects/shared_projects/Data' + "/" + "adhoc" + "_model_rolling_strat.csv", mode = 'a', header = False)
            elif path.exists('/home/brian/Documents/projects/shared_projects/Data' + "/" + "adhoc" + "_model_rolling_strat.csv") == False:
                perf.to_csv('/home/brian/Documents/projects/shared_projects/Data' + "/" + "adhoc" + "_model_rolling_strat.csv", header = True)

##### CREATE FUNCTION #####

def run_ml_backtest(algos, tickers):
    numest = [5,10,50,100,200]
    maxdepth = [1,2,3]
    predwindow = [3,5,10,21,63,126,256]
    





























