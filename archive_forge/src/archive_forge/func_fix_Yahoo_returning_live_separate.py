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
def fix_Yahoo_returning_live_separate(quotes, interval, tz_exchange):
    n = quotes.shape[0]
    if n > 1:
        dt1 = quotes.index[n - 1]
        dt2 = quotes.index[n - 2]
        if quotes.index.tz is None:
            dt1 = dt1.tz_localize('UTC')
            dt2 = dt2.tz_localize('UTC')
        dt1 = dt1.tz_convert(tz_exchange)
        dt2 = dt2.tz_convert(tz_exchange)
        if interval == '1d':
            if dt1.date() == dt2.date():
                quotes = quotes.drop(quotes.index[n - 2])
        else:
            if interval == '1wk':
                last_rows_same_interval = dt1.year == dt2.year and dt1.week == dt2.week
            elif interval == '1mo':
                last_rows_same_interval = dt1.month == dt2.month
            elif interval == '3mo':
                last_rows_same_interval = dt1.year == dt2.year and dt1.quarter == dt2.quarter
            else:
                last_rows_same_interval = dt1 - dt2 < _pd.Timedelta(interval)
            if last_rows_same_interval:
                idx1 = quotes.index[n - 1]
                idx2 = quotes.index[n - 2]
                if idx1 == idx2:
                    return quotes
                if _np.isnan(quotes.loc[idx2, 'Open']):
                    quotes.loc[idx2, 'Open'] = quotes['Open'].iloc[n - 1]
                if not _np.isnan(quotes['High'].iloc[n - 1]):
                    quotes.loc[idx2, 'High'] = _np.nanmax([quotes['High'].iloc[n - 1], quotes['High'].iloc[n - 2]])
                    if 'Adj High' in quotes.columns:
                        quotes.loc[idx2, 'Adj High'] = _np.nanmax([quotes['Adj High'].iloc[n - 1], quotes['Adj High'].iloc[n - 2]])
                if not _np.isnan(quotes['Low'].iloc[n - 1]):
                    quotes.loc[idx2, 'Low'] = _np.nanmin([quotes['Low'].iloc[n - 1], quotes['Low'].iloc[n - 2]])
                    if 'Adj Low' in quotes.columns:
                        quotes.loc[idx2, 'Adj Low'] = _np.nanmin([quotes['Adj Low'].iloc[n - 1], quotes['Adj Low'].iloc[n - 2]])
                quotes.loc[idx2, 'Close'] = quotes['Close'].iloc[n - 1]
                if 'Adj Close' in quotes.columns:
                    quotes.loc[idx2, 'Adj Close'] = quotes['Adj Close'].iloc[n - 1]
                quotes.loc[idx2, 'Volume'] += quotes['Volume'].iloc[n - 1]
                quotes = quotes.drop(quotes.index[n - 1])
    return quotes