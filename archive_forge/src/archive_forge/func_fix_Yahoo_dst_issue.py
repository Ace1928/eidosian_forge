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
def fix_Yahoo_dst_issue(df, interval):
    if interval in ['1d', '1w', '1wk']:
        f_pre_midnight = (df.index.minute == 0) & df.index.hour.isin([22, 23])
        dst_error_hours = _np.array([0] * df.shape[0])
        dst_error_hours[f_pre_midnight] = 24 - df.index[f_pre_midnight].hour
        df.index += _pd.to_timedelta(dst_error_hours, 'h')
    return df