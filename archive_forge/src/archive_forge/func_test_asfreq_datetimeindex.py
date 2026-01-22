from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_datetimeindex(self):
    df = DataFrame({'A': [1, 2, 3]}, index=[datetime(2011, 11, 1), datetime(2011, 11, 2), datetime(2011, 11, 3)])
    df = df.asfreq('B')
    assert isinstance(df.index, DatetimeIndex)
    ts = df['A'].asfreq('B')
    assert isinstance(ts.index, DatetimeIndex)