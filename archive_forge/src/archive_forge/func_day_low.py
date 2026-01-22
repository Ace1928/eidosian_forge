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
def day_low(self):
    if self._day_low is not None:
        return self._day_low
    prices = self._get_1y_prices()
    if prices.empty:
        self._day_low = None
    else:
        self._day_low = float(prices['Low'].iloc[-1])
        if _np.isnan(self._day_low):
            self._day_low = None
    return self._day_low