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
def fix_Yahoo_returning_prepost_unrequested(quotes, interval, tradingPeriods):
    tps_df = tradingPeriods.copy()
    tps_df['_date'] = tps_df.index.date
    quotes['_date'] = quotes.index.date
    idx = quotes.index.copy()
    quotes = quotes.merge(tps_df, how='left')
    quotes.index = idx
    f_drop = quotes.index >= quotes['end']
    f_drop = f_drop | (quotes.index < quotes['start'])
    if f_drop.any():
        quotes = quotes[~f_drop]
    quotes = quotes.drop(['_date', 'start', 'end'], axis=1)
    return quotes