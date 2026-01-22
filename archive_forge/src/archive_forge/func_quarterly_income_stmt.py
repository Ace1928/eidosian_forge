from __future__ import print_function
import datetime as _datetime
from collections import namedtuple as _namedtuple
import pandas as _pd
from .base import TickerBase
from .const import _BASE_URL_
@property
def quarterly_income_stmt(self) -> _pd.DataFrame:
    return self.get_income_stmt(pretty=True, freq='quarterly')