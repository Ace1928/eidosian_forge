import pandas as pd  # For data manipulation and analysis
import requests  # For making HTTP requests to a specified URL
import pandas_ta as ta  # For technical analysis indicators
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
from termcolor import colored as cl  # For coloring terminal text
import math  # Provides access to mathematical functions
import logging  # For tracking events that happen when some software runs
import nltk  # For natural language processing tasks
from newspaper import Article  # For extracting and parsing news articles
import ccxt  # Cryptocurrency exchange library for connecting to various exchanges
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Union, Optional, Tuple, List, Dict, Any, Callable
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import gspread
from oauth2client.client import OAuth2Credentials
from tkinter import Tk, Label, Entry, Button, StringVar
import json
import os
import yfinance as yf
from itertools import product
from concurrent.futures import ThreadPoolExecutor, wait
import time
import yaml
import traceback
from typing import TypeAlias
@log_exception
@log_function_call
def get_historical_data(symbol: str, start_date: str, end_date: str, interval: str, data_source: str='google_sheets', sheet_url: Optional[str]=None, benzinga_api_key: Optional[str]=None, yahoo_api_key: Optional[str]=None) -> pd.DataFrame:
    """
    Fetches historical stock data from the specified data source (Google Sheets, Benzinga API, or Yahoo Finance API).

    Parameters:
    - symbol (str): The stock symbol to fetch historical data for.
    - start_date (str): The start date for the historical data in YYYY-MM-DD format.
    - end_date (str): The end date for the historical data in YYYY-MM-DD format.
    - interval (str): The interval for the historical data (e.g., "1d" for daily).
    - data_source (str): The data source to fetch the data from. Options: "google_sheets", "benzinga", "yahoo". Default is "google_sheets".
    - sheet_url (str, optional): The URL of the Google Sheets spreadsheet containing the stock data. Required if data_source is "google_sheets".
    - benzinga_api_key (str, optional): The Benzinga API key. Required if data_source is "benzinga".
    - yahoo_api_key (str, optional): The Yahoo Finance API key. Required if data_source is "yahoo".

    Returns:
    - pd.DataFrame: A DataFrame containing the historical stock data.

    Raises:
    - ValueError: If the required parameters for the selected data source are not provided.
    - Exception: If there is an error fetching data from the specified data source.

    Example:
    >>> get_historical_data("AAPL", "2022-01-01", "2023-06-08", "1d", data_source="google_sheets", sheet_url="https://docs.google.com/spreadsheets/d/.../edit#gid=0")
    """
    try:
        if data_source == 'google_sheets':
            if sheet_url is None:
                raise ValueError("sheet_url is required when data_source is 'google_sheets'")
            df = pd.read_csv(sheet_url.replace('/edit#gid=', '/export?format=csv&gid='))
            df.columns = ['date', 'close']
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']
            df['volume'] = 0
        elif data_source == 'benzinga':
            if benzinga_api_key is None:
                raise ValueError("benzinga_api_key is required when data_source is 'benzinga'")
            url = 'https://api.benzinga.com/api/v2/bars'
            querystring = {'token': benzinga_api_key, 'symbols': symbol, 'from': start_date, 'to': end_date, 'interval': interval}
            response = requests.get(url, params=querystring)
            response.raise_for_status()
            hist_json = response.json()
            if not (hist_json and isinstance(hist_json, list) and ('candles' in hist_json[0])):
                raise ValueError('Unexpected JSON structure received from the Benzinga API.')
            df = pd.DataFrame(hist_json[0]['candles'])
        elif data_source == 'yahoo':
            if yahoo_api_key is None:
                raise ValueError("yahoo_api_key is required when data_source is 'yahoo'")
            url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
            querystring = {'period1': int(pd.to_datetime(start_date).timestamp()), 'period2': int(pd.to_datetime(end_date).timestamp()), 'interval': interval}
            headers = {'x-api-key': yahoo_api_key}
            response = requests.get(url, params=querystring, headers=headers)
            response.raise_for_status()
            data = response.json()
            if not (data and 'chart' in data and ('result' in data['chart']) and data['chart']['result']):
                raise ValueError('Unexpected JSON structure received from the Yahoo Finance API.')
            df = pd.DataFrame(data['chart']['result'][0]['indicators']['quote'][0])
            df.index = pd.to_datetime(data['chart']['result'][0]['timestamp'], unit='s')
            df.index.name = 'date'
            df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
        else:
            raise ValueError(f'Unsupported data source: {data_source}')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        logging.error(f'Error fetching historical data from {data_source}: {e}')
        raise