import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import ccxt  # For connecting to various trading exchanges
import backtrader as bt  # For backtesting
import asyncio
import aiohttp
import websocket
import logging
import yfinance as yf  # For downloading market data from Yahoo Finance
def backtest_with_costs(self, strategy) -> None:
    """Backtest a given strategy including transaction costs."""
    logging.info('Backtesting strategy with transaction costs.')
    transaction_costs = 0.1
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy)
    data = bt.feeds.PandasData(dataname=self.data)
    cerebro.adddata(data)
    cerebro.run()
    self.data['net_profit'] = self.data['profit'] - self.data['trade_amount'] * transaction_costs / 100
    cerebro.plot()
    logging.info('Backtesting complete.')