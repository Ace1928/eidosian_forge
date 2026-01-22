from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_grouper_period_index(self):
    periods = 2
    index = pd.period_range(start='2018-01', periods=periods, freq='M', name='Month')
    period_series = Series(range(periods), index=index)
    result = period_series.groupby(period_series.index.month).sum()
    expected = Series(range(periods), index=Index(range(1, periods + 1), name=index.name))
    tm.assert_series_equal(result, expected)