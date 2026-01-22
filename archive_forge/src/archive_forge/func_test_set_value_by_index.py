import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_value_by_index(self):
    warn = None
    msg = 'will attempt to set the values inplace'
    df = DataFrame(np.arange(9).reshape(3, 3).T)
    df.columns = list('AAA')
    expected = df.iloc[:, 2].copy()
    with tm.assert_produces_warning(warn, match=msg):
        df.iloc[:, 0] = 3
    tm.assert_series_equal(df.iloc[:, 2], expected)
    df = DataFrame(np.arange(9).reshape(3, 3).T)
    df.columns = [2, float(2), str(2)]
    expected = df.iloc[:, 1].copy()
    with tm.assert_produces_warning(warn, match=msg):
        df.iloc[:, 0] = 3
    tm.assert_series_equal(df.iloc[:, 1], expected)