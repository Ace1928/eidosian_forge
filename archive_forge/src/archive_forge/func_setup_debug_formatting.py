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
def setup_debug_formatting():
    global yf_logger
    yf_logger = get_yf_logger()
    if not yf_logger.isEnabledFor(logging.DEBUG):
        yf_logger.warning("logging mode not set to 'DEBUG', so not setting up debug formatting")
        return
    global yf_log_indented
    if not yf_log_indented:
        if yf_logger.handlers is None or len(yf_logger.handlers) == 0:
            h = logging.StreamHandler()
            formatter = MultiLineFormatter(fmt='%(levelname)-8s %(message)s')
            h.setFormatter(formatter)
            yf_logger.addHandler(h)
    yf_log_indented = True