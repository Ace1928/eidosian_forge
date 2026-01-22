import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import _BASE_URL_
from yfinance.exceptions import YFinanceDataException
@property
def insider_purchases(self) -> pd.DataFrame:
    if self._insider_purchases is None:
        self._fetch_and_parse()
    return self._insider_purchases