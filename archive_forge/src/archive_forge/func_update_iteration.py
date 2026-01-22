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
def update_iteration(self, val=None):
    val = val if val is not None else self.elapsed / float(self.iterations)
    self.__update_amount(val * 100.0)
    self.prog_bar += f'  {self.elapsed} of {self.iterations} {self.text}'