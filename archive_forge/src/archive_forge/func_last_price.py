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
def last_price(self):
    if self._last_price is not None:
        return self._last_price
    prices = self._get_1y_prices()
    if prices.empty:
        md = self._get_exchange_metadata()
        if 'regularMarketPrice' in md:
            self._last_price = md['regularMarketPrice']
    else:
        self._last_price = float(prices['Close'].iloc[-1])
        if _np.isnan(self._last_price):
            md = self._get_exchange_metadata()
            if 'regularMarketPrice' in md:
                self._last_price = md['regularMarketPrice']
    return self._last_price