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
def get_news(self, proxy=None) -> list:
    if self._news:
        return self._news
    url = f'{_BASE_URL_}/v1/finance/search?q={self.ticker}'
    data = self._data.cache_get(url=url, proxy=proxy)
    if 'Will be right back' in data.text:
        raise RuntimeError('*** YAHOO! FINANCE IS CURRENTLY DOWN! ***\nOur engineers are working quickly to resolve the issue. Thank you for your patience.')
    data = data.json()
    self._news = data.get('news', [])
    return self._news