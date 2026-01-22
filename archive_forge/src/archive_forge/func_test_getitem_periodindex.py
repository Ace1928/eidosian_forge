import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_periodindex(self):
    rng = period_range('1/1/2000', periods=5)
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)), columns=rng)
    ts = df[rng[0]]
    tm.assert_series_equal(ts, df.iloc[:, 0])
    ts = df['1/1/2000']
    tm.assert_series_equal(ts, df.iloc[:, 0])