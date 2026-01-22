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
def load_credentials_gui() -> str:
    """
    Opens a GUI window for the user to input Google Sheets API credentials and saves them to a JSON file.

    Returns:
    - str: The path to the saved credentials JSON file.

    Example:
    >>> credentials_path = load_credentials_gui()
    """

    def save_credentials() -> None:
        credentials = {'installed': {'client_id': client_id.get(), 'project_id': project_id.get(), 'auth_uri': 'https://accounts.google.com/o/oauth2/auth', 'token_uri': 'https://oauth2.googleapis.com/token', 'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs', 'client_secret': client_secret.get(), 'redirect_uris': ['http://localhost']}}
        with open('credentials.json', 'w') as file:
            json.dump(credentials, file)
        root.destroy()
    root = Tk()
    root.title('Google Sheets API Credentials')
    project_id = StringVar()
    client_id = StringVar()
    client_secret = StringVar()
    Label(root, text='Project ID:').grid(row=0, column=0, sticky='e')
    Entry(root, textvariable=project_id).grid(row=0, column=1)
    Label(root, text='Client ID:').grid(row=1, column=0, sticky='e')
    Entry(root, textvariable=client_id).grid(row=1, column=1)
    Label(root, text='Client Secret:').grid(row=2, column=0, sticky='e')
    Entry(root, textvariable=client_secret, show='*').grid(row=2, column=1)
    Button(root, text='Save Credentials', command=save_credentials).grid(row=3, column=0, columnspan=2, pady=10)
    root.mainloop()
    return 'credentials.json'