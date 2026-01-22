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
def log_function_call(func: Callable) -> Callable:
    """
    Decorator to log function calls.

    Purpose:
        This decorator logs the entry and exit of a function, providing insights into the function's usage
        and aiding in debugging and monitoring.

    Parameters:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.

    Example:
        @log_function_call
        def example_function():
            pass
    """

    def wrapper(*args, **kwargs) -> Any:
        logging.info(f'Entering {func.__name__}')
        result: Any = func(*args, **kwargs)
        logging.info(f'Exiting {func.__name__}')
        return result
    return wrapper