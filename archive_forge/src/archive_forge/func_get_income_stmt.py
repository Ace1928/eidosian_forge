from __future__ import print_function
from io import StringIO
import json as _json
import warnings
from typing import Optional, Union
from urllib.parse import quote as urlencode
import pandas as pd
import requests
from . import utils, cache
from .data import YfData
from .scrapers.analysis import Analysis
from .scrapers.fundamentals import Fundamentals
from .scrapers.holders import Holders
from .scrapers.quote import Quote, FastInfo
from .scrapers.history import PriceHistory
from .const import _BASE_URL_, _ROOT_URL_
def get_income_stmt(self, proxy=None, as_dict=False, pretty=False, freq='yearly'):
    """
        :Parameters:
            as_dict: bool
                Return table as Python dict
                Default is False
            pretty: bool
                Format row names nicely for readability
                Default is False
            freq: str
                "yearly" or "quarterly"
                Default is "yearly"
            proxy: str
                Optional. Proxy server URL scheme
                Default is None
        """
    self._fundamentals.proxy = proxy or self.proxy
    data = self._fundamentals.financials.get_income_time_series(freq=freq, proxy=proxy)
    if pretty:
        data = data.copy()
        data.index = utils.camel2title(data.index, sep=' ', acronyms=['EBIT', 'EBITDA', 'EPS', 'NI'])
    if as_dict:
        return data.to_dict()
    return data