import pandas as pd  # Data manipulation
import requests  # HTTP requests
import pandas_ta as ta  # Technical analysis
import matplotlib as mpl  # Plotting
import matplotlib.pyplot as plt  # Plotting
from termcolor import colored as cl  # Text customization
import math  # Mathematical operations
import numpy as np  # Numerical operations
from datetime import datetime as dt  # Date and time operations
from typing import (
import sqlite3  # Database operations
import yfinance as yf  # Yahoo Finance API
from sqlite3 import Connection, Cursor
from typing import Optional  # Type hinting
import seaborn as sns  # Data visualization
import logging  # Logging
import time  # Time operations
import sys  # System-specific parameters and functions
from scripts.trading_bot.indecache import async_cache  # Async cache decorator
def fetch_stock_data(ticker: str, source: str, interval: str='1d', start_date: str='2000-01-01', end_date: str=dt.now().strftime('%Y-%m-%d')) -> pd.DataFrame:
    """
    Fetches historical stock data from various APIs based on the provided parameters.

    Parameters:
    - ticker (str): The stock ticker symbol to fetch data for.
    - source (str): The API source to fetch data from (e.g., 'yahoo', 'alphavantage', 'iex', 'quandl', 'finage', 'twelvedata', 'polygon', 'finnhub').
    - interval (str): The interval of the stock data (default is '1d' for daily data).
    - start_date (str): The start date of the historical data in 'YYYY-MM-DD' format (default is '2000-01-01').
    - end_date (str): The end date of the historical data in 'YYYY-MM-DD' format (default is the current date).

    Returns:
    - Optional[DataFrame]: A DataFrame containing the fetched stock data, or None if no data is fetched.
    """
    data: pd.DataFrame = None
    db = StockDatabase(DB_PATH)
    if source == 'yahoo':
        data = yf.download(ticker, start='1970-01-01', end=dt.now().strftime('%Y-%m-%d'), interval='1d')
    elif source == 'alphavantage':
        api_key: str = 'YOUR_ALPHA_VANTAGE_API_KEY'
        url: str = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&outputsize=full'
        r: requests.Response = requests.get(url)
        if r.status_code == 200:
            data = pd.DataFrame(r.json()['Time Series (Daily)']).transpose().rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
        else:
            print(f'Error fetching data from Alpha Vantage. Status code: {r.status_code}')
    elif source == 'iex':
        api_token: str = 'YOUR_IEX_API_TOKEN'
        url: str = f'https://cloud.iexapis.com/stable/stock/{ticker}/chart/5y?token={api_token}'
        r: requests.Response = requests.get(url)
        if r.status_code == 200:
            data = pd.read_json(url)
        else:
            print(f'Error fetching data from IEX. Status code: {r.status_code}')
    elif source == 'quandl':
        api_key: str = 'YOUR_QUANDL_API_KEY'
        url: str = f'https://www.quandl.com/api/v3/datasets/WIKI/{ticker}.json?start_date={start_date}&end_date={end_date}&api_key={api_key}'
        r: requests.Response = requests.get(url)
        if r.status_code == 200:
            data = pd.DataFrame(r.json()['dataset']['data'], columns=r.json()['dataset']['column_names']).set_index('Date')
        else:
            print(f'Error fetching data from Quandl. Status code: {r.status_code}')
    elif source == 'finage':
        api_key: str = 'YOUR_FINAGE_API_KEY'
        url: str = f'https://api.finage.co.uk/last/stock/{ticker}?apikey={api_key}'
        r: requests.Response = requests.get(url)
        if r.status_code == 200:
            data = pd.DataFrame([r.json()], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            data['date'] = dt.now().strftime('%Y-%m-%d')
        else:
            print(f'Error fetching data from Finage. Status code: {r.status_code}')
    elif source == 'twelvedata':
        api_key: str = 'YOUR_TWELVEDATA_API_KEY'
        url: str = f'https://api.twelvedata.com/time_series?symbol={ticker}&interval={interval}&apikey={api_key}&start_date={start_date}&end_date={end_date}'
        r: requests.Response = requests.get(url)
        if r.status_code == 200:
            data = pd.DataFrame(r.json()['values']).rename(columns={'datetime': 'date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}).set_index('date')
        else:
            print(f'Error fetching data from TwelveData. Status code: {r.status_code}')
    elif source == 'polygon':
        api_key: str = 'YOUR_POLYGON_API_KEY'
        url: str = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={api_key}'
        r: requests.Response = requests.get(url)
        if r.status_code == 200:
            data = pd.DataFrame(r.json()['results'], columns=['t', 'o', 'h', 'l', 'c', 'v'])
            data.rename(columns={'t': 'date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
            data['date'] = pd.to_datetime(data['date'], unit='ms').dt.strftime('%Y-%m-%d')
        else:
            print(f'Error fetching data from Polygon. Status code: {r.status_code}')
    elif source == 'finnhub':
        api_key: str = 'YOUR_FINNHUB_API_KEY'
        url: str = f'https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=D&from={int(dt.datetime.strptime(start_date, '%Y-%m-%d').timestamp())}&to={int(dt.datetime.strptime(end_date, '%Y-%m-%d').timestamp())}&token={api_key}'
        r: requests.Response = requests.get(url)
        if r.status_code == 200:
            json_data = r.json()
            data = pd.DataFrame({'date': pd.to_datetime(json_data['t'], unit='s'), 'open': json_data['o'], 'high': json_data['h'], 'low': json_data['l'], 'close': json_data['c'], 'volume': json_data['v']})
            data.set_index('date', inplace=True)
        else:
            print(f'Error fetching data from Finnhub. Status code: {r.status_code}')
    elif source == 'tiingo':
        api_key: str = 'YOUR_TIINGO_API_KEY'
        url: str = f'https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={api_key}'
        r: requests.Response = requests.get(url)
        if r.status_code == 200:
            data = pd.DataFrame(r.json())
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
        else:
            print(f'Error fetching data from Tiingo. Status code: {r.status_code}')
    elif source == 'eodhistoricaldata':
        api_key: str = 'YOUR_EODHISTORICALDATA_API_KEY'
        url: str = f'https://eodhistoricaldata.com/api/eod/{ticker}?from={start_date}&to={end_date}&api_token={api_key}'
        r: requests.Response = requests.get(url)
        if r.status_code == 200:
            data = pd.DataFrame(r.json())
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
        else:
            print(f'Error fetching data from EODHistoricalData. Status code: {r.status_code}')
    elif source == 'marketstack':
        api_key: str = 'YOUR_MARKETSTACK_API_KEY'
        url: str = f'http://api.marketstack.com/v1/eod?access_key={api_key}&symbols={ticker}&date_from={start_date}&date_to={end_date}'
        r: requests.Response = requests.get(url)
        if r.status_code == 200:
            data = pd.DataFrame(r.json()['data'])
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
        else:
            print(f'Error fetching data from Marketstack. Status code: {r.status_code}')
    elif source == 'alpaca':
        api_key: str = 'YOUR_ALPACA_API_KEY'
        secret_key: str = 'YOUR_ALPACA_SECRET_KEY'
        headers: Dict[str, str] = {'APCA-API-KEY-ID': api_key, 'APCA-API-SECRET-KEY': secret_key}
        url: str = f'https://data.alpaca.markets/v2/stocks/{ticker}/bars?start={start_date}&end={end_date}&timeframe={interval}'
        r: requests.Response = requests.get(url, headers=headers)
        if r.status_code == 200:
            data = pd.DataFrame(r.json()['bars'])
            data['date'] = pd.to_datetime(data['t'])
            data.set_index('date', inplace=True)
            data.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        else:
            print(f'Error fetching data from Alpaca. Status code: {r.status_code}')
    else:
        raise ValueError(f'Unsupported data source: {source}')
    if data is not None:
        data = data.reset_index()
        data['ticker'] = ticker
        data = data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        data = data[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
        data = data.to_dict(orient='records')
        db = StockDatabase(DB_PATH)
        for row in data:
            db.update_table('Market_Data', row)
    else:
        print('No data fetched.')
    return data