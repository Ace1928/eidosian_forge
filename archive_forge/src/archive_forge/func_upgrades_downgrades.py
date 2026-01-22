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
def upgrades_downgrades(self) -> pd.DataFrame:
    if self._upgrades_downgrades is None:
        result = self._fetch(self.proxy, modules=['upgradeDowngradeHistory'])
        if result is None:
            self._upgrades_downgrades = pd.DataFrame()
        else:
            try:
                data = result['quoteSummary']['result'][0]['upgradeDowngradeHistory']['history']
                if len(data) == 0:
                    raise YFinanceDataException(f'No upgrade/downgrade history found for {self._symbol}')
                df = pd.DataFrame(data)
                df.rename(columns={'epochGradeDate': 'GradeDate', 'firm': 'Firm', 'toGrade': 'ToGrade', 'fromGrade': 'FromGrade', 'action': 'Action'}, inplace=True)
                df.set_index('GradeDate', inplace=True)
                df.index = pd.to_datetime(df.index, unit='s')
                self._upgrades_downgrades = df
            except (KeyError, IndexError):
                raise YFinanceDataException(f'Failed to parse json response from Yahoo Finance: {result}')
    return self._upgrades_downgrades