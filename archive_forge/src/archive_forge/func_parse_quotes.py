from __future__ import print_function
import datetime as _datetime
import logging
import re as _re
import sys as _sys
import threading
from functools import lru_cache
from inspect import getmembers
from types import FunctionType
from typing import List, Optional
import numpy as _np
import pandas as _pd
import pytz as _tz
import requests as _requests
from dateutil.relativedelta import relativedelta
from pytz import UnknownTimeZoneError
from yfinance import const
from .const import _BASE_URL_
def parse_quotes(data):
    timestamps = data['timestamp']
    ohlc = data['indicators']['quote'][0]
    volumes = ohlc['volume']
    opens = ohlc['open']
    closes = ohlc['close']
    lows = ohlc['low']
    highs = ohlc['high']
    adjclose = closes
    if 'adjclose' in data['indicators']:
        adjclose = data['indicators']['adjclose'][0]['adjclose']
    quotes = _pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes, 'Adj Close': adjclose, 'Volume': volumes})
    quotes.index = _pd.to_datetime(timestamps, unit='s')
    quotes.sort_index(inplace=True)
    return quotes