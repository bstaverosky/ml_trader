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
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from os import path
import pyfolio as pf
import yfinance as yf
import talib as talib
import time

##### PARAMETERS #####

# Delete current database?
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

# Tickers already run
#tkran = pd.read_csv("/home/bstaverosky/Documents/projects/python/data/ETF_universe_xgboost_rolling_strat.csv")
#tkran = tkran.loc[:,"Ticker"].tolist()

#tickers = list(set(tickers) - set(tkran))

#exclude = ["FLTR","MTGP","FLFR","FLUD"]

#tickers = list(set(tickers) - set(exclude))

for x in tickers:
    #time.sleep(60)
    #print(x)
    asset = yf.download(x, start='1900-01-01', progress=True)
    #asset = assets.iloc[:, assets.columns.get_level_values(1)==x]
    #asset.columns = asset.columns.droplevel(1)
    
    if len(asset.index) < 252:    

        # Performance Data Frame
        perf = pd.DataFrame({
            'Date Run': datetime.today().strftime('%Y-%m-%d'),
            'Ticker': x,
            'Prediction Window': [0],
            'Lookback Window': [0],
            'Number of Estimators': [0],
            'Max Depth': [0],
            'Annual Return': [0],
            'Cumulative Returns': [0],
            'Sharpe Ratio': [0],
            'Sortino Ratio': [0],
            'Max Drawdown': [0],
            'Mean Squared Error': [0],
            'Baseline': [0],
            'Accuracy': [0],
            'Skill': [0]
        })
        
        # Save to CSV
        if path.exists('/home/brian/Documents/projects/shared_projects/Data' + "/" + "ETF_universe" + "_xgboost_rolling_strat.csv") == True:
            perf.to_csv('/home/brian/Documents/projects/shared_projects/Data' + "/" + "ETF_universe" + "_xgboost_rolling_strat.csv", mode = 'a', header = False)
        elif path.exists('/home/brian/Documents/projects/shared_projects/Data' + "/" + "ETF_universe" + "_xgboost_rolling_strat.csv") == False:
            perf.to_csv('/home/brian/Documents/projects/shared_projects/Data' + "/" + "ETF_universe" + "_xgboost_rolling_strat.csv", header = True)
            
    elif len(asset.index) >= 252:
    
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
        #test = asset.resample('W').ffill()
        asset['closelag5'] = asset['Close'].shift(pw)
        def percentage_change(col1,col2):
            return ((col2 - col1) / col1) * 100
        
        asset['pwret'] = percentage_change(asset['closelag5'],asset['Close'])
        asset['pwret'] = asset['pwret'].shift(-pw-1)
        
        # CLEAN DATAFRAME
        df = asset[['sma_rat', 'vme_rat', 'vol_rat', 'p2h', 'pwret']]
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
        df = df.dropna()
        
        if len(df.index) < 252:
        
            print("ticker does not have enough history")    
        
        elif len(df.index) >= 252 :
            #j = lw
            predf = pd.DataFrame(columns = ["pred"])
            
            # Create rolling window trainset
            for i in range((lw+pw), len(df.index)-1):
                gb = GradientBoostingRegressor(n_estimators = nest,
                                               max_depth=mdepth,
                                               random_state=2)
                # Make trainsets
                #xtrain = df.loc[df.index[i-(lw+pw)]:df.index[i-pw],['sma_rat', 'vol_rat', 'p2h', 'vme_rat']]
                #ytrain = df.loc[df.index[i-(lw+pw)]:df.index[i-pw],['pwret']]
                
                # Make testsets
                #xtest = df.loc[[df.index[i+1]],['sma_rat', 'vol_rat', 'p2h','vme_rat']]    
                #ytest = df.loc[[df.index[i+1]],['pwret']]
                #type(ytest)
                
                # Make trainsets
                xtrain = df.loc[df.index[i-(lw+pw)]:df.index[i-pw],['sma_rat', 'vol_rat', 'p2h']]
                ytrain = df.loc[df.index[i-(lw+pw)]:df.index[i-pw],['pwret']]
                    
                # Make testsets
                xtest = df.loc[[df.index[i+1]],['sma_rat', 'vol_rat', 'p2h']]    
                ytest = df.loc[[df.index[i+1]],['pwret']]
                
                gb.fit(xtrain, ytrain)
                y_pred = gb.predict(xtest)

                
                #results = pd.DataFrame(data = )
                
                lframe = pd.DataFrame(y_pred, columns = ["pred"], index = ytest.index)
                predf = predf.append(lframe)
                
            # Put predictions back on original data frame
            # And convert y_pred so it can be added to dataframe
            sframe = df
            sframe['signal'] = predf
            sframe['signal'] = sframe['signal'].shift(1)
            #sframe['signal'] = sframe['signal']
            sframe['return'] = asset['dayret']
            
            if len(sframe) < 5:
                print("no strategy history")
                
            elif len(sframe) > 5:
            
                # Create the strategy return performance
                for i in range(len(sframe.index)):
                    if sframe.loc[sframe.index[i], "signal"] > 0:
                        sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*1.25
                    else:
                        sframe.loc[sframe.index[i], "strat"] = sframe.loc[sframe.index[i], "return"]*0.75
                        
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
                                
                
                # Performance Data Frame
                # Performance Data Frame
                perf = pd.DataFrame({
                    'Date Run': datetime.today().strftime('%Y-%m-%d'),
                    'Ticker': x,
                    'Prediction Window': [pw],
                    'Lookback Window': [lw],
                    'Number of Estimators': nest,
                    'Max Depth': mdepth,
                    'Annual Return': ep.cagr(strat_series),
                    'Cumulative Returns': ep.cum_returns_final(strat_series)*100,
                    'Sharpe Ratio': ep.sharpe_ratio(strat_series),
                    'Sortino Ratio': ep.sortino_ratio(strat_series),
                    'Max Drawdown': ep.max_drawdown(strat_series),
                    'Mean Squared Error': MSE(sframe.loc[:,"pwret"], sframe.loc[:,"signal"])**(1/2),
                    'Baseline': (sframe.loc[:,"pwret_bin"] == 1).sum()/len(sframe),
                    'Accuracy': accuracy_score(sframe.loc[:,"pwret_bin"], sframe.loc[:,"signal_bin"]),
                    'Skill': accuracy_score(sframe.loc[:,"pwret_bin"], sframe.loc[:,"signal_bin"])-(sframe.loc[:,"pwret_bin"] == 1).sum()/len(sframe)
                })

                # Save to CSV
                if path.exists('/home/brian/Documents/projects/shared_projects/Data' + "/" + "multi_etf" + "_xgboost_rolling_strat.csv") == True:
                    perf.to_csv('/home/brian/Documents/projects/shared_projects/Data' + "/" + "multi_etf" + "_xgboost_rolling_strat.csv", mode = 'a', header = False)
                elif path.exists('/home/brian/Documents/projects/shared_projects/Data' + "/" + "multi_etf" + "_xgboost_rolling_strat.csv") == False:
                    perf.to_csv('/home/brian/Documents/projects/shared_projects/Data' + "/" + "multi_etf" + "_xgboost_rolling_strat.csv", header = True)
            
        
    #pf.create_simple_tear_sheet(returns = strat_series, benchmark_rets=bmk_series)
    
#sframe['signal'].describe()
#rtpred = asset.loc[[asset.index[len(asset)-1]], ['sma_rat', 'vol_rat', 'p2h', 'vme_rat']]
#gb.predict(rtpred)




