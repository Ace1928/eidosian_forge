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
def format_history_metadata(md, tradingPeriodsOnly=True):
    if not isinstance(md, dict):
        return md
    if len(md) == 0:
        return md
    tz = md['exchangeTimezoneName']
    if not tradingPeriodsOnly:
        for k in ['firstTradeDate', 'regularMarketTime']:
            if k in md and md[k] is not None:
                if isinstance(md[k], int):
                    md[k] = _pd.to_datetime(md[k], unit='s', utc=True).tz_convert(tz)
        if 'currentTradingPeriod' in md:
            for m in ['regular', 'pre', 'post']:
                if m in md['currentTradingPeriod'] and isinstance(md['currentTradingPeriod'][m]['start'], int):
                    for t in ['start', 'end']:
                        md['currentTradingPeriod'][m][t] = _pd.to_datetime(md['currentTradingPeriod'][m][t], unit='s', utc=True).tz_convert(tz)
                    del md['currentTradingPeriod'][m]['gmtoffset']
                    del md['currentTradingPeriod'][m]['timezone']
    if 'tradingPeriods' in md:
        tps = md['tradingPeriods']
        if tps == {'pre': [], 'post': []}:
            pass
        elif isinstance(tps, (list, dict)):
            if isinstance(tps, list):
                df = _pd.DataFrame.from_records(_np.hstack(tps))
                df = df.drop(['timezone', 'gmtoffset'], axis=1)
                df['start'] = _pd.to_datetime(df['start'], unit='s', utc=True).dt.tz_convert(tz)
                df['end'] = _pd.to_datetime(df['end'], unit='s', utc=True).dt.tz_convert(tz)
            elif isinstance(tps, dict):
                pre_df = _pd.DataFrame.from_records(_np.hstack(tps['pre']))
                post_df = _pd.DataFrame.from_records(_np.hstack(tps['post']))
                regular_df = _pd.DataFrame.from_records(_np.hstack(tps['regular']))
                pre_df = pre_df.rename(columns={'start': 'pre_start', 'end': 'pre_end'}).drop(['timezone', 'gmtoffset'], axis=1)
                post_df = post_df.rename(columns={'start': 'post_start', 'end': 'post_end'}).drop(['timezone', 'gmtoffset'], axis=1)
                regular_df = regular_df.drop(['timezone', 'gmtoffset'], axis=1)
                cols = ['pre_start', 'pre_end', 'start', 'end', 'post_start', 'post_end']
                df = regular_df.join(pre_df).join(post_df)
                for c in cols:
                    df[c] = _pd.to_datetime(df[c], unit='s', utc=True).dt.tz_convert(tz)
                df = df[cols]
            df.index = _pd.to_datetime(df['start'].dt.date)
            df.index = df.index.tz_localize(tz)
            df.index.name = 'Date'
            md['tradingPeriods'] = df
    return md