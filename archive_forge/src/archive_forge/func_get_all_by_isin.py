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
def get_all_by_isin(isin, proxy=None, session=None):
    if not is_isin(isin):
        raise ValueError('Invalid ISIN number')
    session = session or _requests
    url = f'{_BASE_URL_}/v1/finance/search?q={isin}'
    data = session.get(url=url, proxies=proxy, headers=user_agent_headers)
    try:
        data = data.json()
        ticker = data.get('quotes', [{}])[0]
        return {'ticker': {'symbol': ticker['symbol'], 'shortname': ticker['shortname'], 'longname': ticker['longname'], 'type': ticker['quoteType'], 'exchange': ticker['exchDisp']}, 'news': data.get('news', [])}
    except Exception:
        return {}