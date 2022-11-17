#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:47:33 2022

@author: brian
"""

def percentage_change(col1,col2):
    return ((col2 - col1) / col1) * 100


def make_perf_output(ticker,strat_series,lw,pw,nest,mdepth,sframe,min_length):
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
                
    elif min_length == True:
        perf = pd.DataFrame({
            'Date Run': datetime.today().strftime('%Y-%m-%d'),
            'Ticker': ticker,
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
        return (perf)