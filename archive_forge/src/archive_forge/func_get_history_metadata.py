import datetime as _datetime
import dateutil as _dateutil
import logging
import numpy as np
import pandas as pd
import time as _time
from yfinance import shared, utils
from yfinance.const import _BASE_URL_, _PRICE_COLNAMES_
def get_history_metadata(self, proxy=None) -> dict:
    if self._history_metadata is None:
        self.history(period='1wk', interval='1h', prepost=True, proxy=proxy)
    if self._history_metadata_formatted is False:
        self._history_metadata = utils.format_history_metadata(self._history_metadata)
        self._history_metadata_formatted = True
    return self._history_metadata