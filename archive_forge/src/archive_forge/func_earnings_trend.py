import pandas as pd
from yfinance.data import YfData
from yfinance.exceptions import YFNotImplementedError
@property
def earnings_trend(self) -> pd.DataFrame:
    if self._earnings_trend is None:
        raise YFNotImplementedError('earnings_trend')
    return self._earnings_trend