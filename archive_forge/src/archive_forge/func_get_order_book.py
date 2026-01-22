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
def get_order_book(exchange_id: str, symbol: str) -> Optional[Dict[str, List[List[float]]]]:
    """

    Fetches the real-time order book data for a given symbol from a specified exchange.

    Parameters:

    - exchange_id (str): Identifier for the exchange (e.g., 'binance').

    - symbol (str): Trading pair symbol (e.g., 'BTC/USD').

    Returns:

    - Optional[Dict[str, List[List[float]]]]: Order book data containing bids and asks, or None if an error occurs.

    Example:

    >>> get_order_book('binance', 'BTC/USD')

    """
    try:
        exchange_class = getattr(ccxt, exchange_id)()
        exchange_class.load_markets()
        order_book = exchange_class.fetch_order_book(symbol)
        return order_book
    except Exception as e:
        logging.error(f'Failed to fetch order book for {symbol} on {exchange_id}: {e}')
        return None