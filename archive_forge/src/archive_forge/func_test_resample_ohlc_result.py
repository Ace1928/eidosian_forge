from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
def test_resample_ohlc_result(unit):
    index = date_range('1-1-2000', '2-15-2000', freq='h').as_unit(unit)
    index = index.union(date_range('4-15-2000', '5-15-2000', freq='h').as_unit(unit))
    s = Series(range(len(index)), index=index)
    a = s.loc[:'4-15-2000'].resample('30min').ohlc()
    assert isinstance(a, DataFrame)
    b = s.loc[:'4-14-2000'].resample('30min').ohlc()
    assert isinstance(b, DataFrame)