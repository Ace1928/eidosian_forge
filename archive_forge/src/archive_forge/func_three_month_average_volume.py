import datetime
import json
import warnings
from collections.abc import MutableMapping
import numpy as _np
import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import quote_summary_valid_modules, _BASE_URL_
from yfinance.exceptions import YFNotImplementedError, YFinanceDataException, YFinanceException
@property
def three_month_average_volume(self):
    if self._3mo_avg_vol is not None:
        return self._3mo_avg_vol
    prices = self._get_1y_prices(fullDaysOnly=True)
    if prices.empty:
        self._3mo_avg_vol = None
    else:
        dt1 = prices.index[-1]
        dt0 = dt1 - utils._interval_to_timedelta('3mo') + utils._interval_to_timedelta('1d')
        self._3mo_avg_vol = int(prices.loc[dt0:dt1, 'Volume'].mean())
    return self._3mo_avg_vol