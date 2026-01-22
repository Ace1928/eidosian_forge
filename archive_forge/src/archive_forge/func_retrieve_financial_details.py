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
def retrieve_financial_details(data):
    """
    retrieve_financial_details returns all of the available financial details under the
    "QuoteTimeSeriesStore" for any of the following three yahoo finance webpages:
    "/financials", "/cash-flow" and "/balance-sheet".

    Returns:
        - TTM_dicts: A dictionary full of all of the available Trailing Twelve Month figures, this can easily be converted to a pandas dataframe.
        - Annual_dicts: A dictionary full of all of the available Annual figures, this can easily be converted to a pandas dataframe.
    """
    TTM_dicts = []
    Annual_dicts = []
    for key, timeseries in data.get('timeSeries', {}).items():
        try:
            if timeseries:
                time_series_dict = {'index': key}
                for each in timeseries:
                    if not each:
                        continue
                    time_series_dict[each.get('asOfDate')] = each.get('reportedValue')
                if 'trailing' in key:
                    TTM_dicts.append(time_series_dict)
                elif 'annual' in key:
                    Annual_dicts.append(time_series_dict)
        except KeyError as e:
            print(f'An error occurred while processing the key: {e}')
    return (TTM_dicts, Annual_dicts)