import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_by_key_by_name(self):
    df = DataFrame({'a': np.array([0, 3, np.nan, 3, 2, np.nan]), 'b': np.array([0, 2, np.nan, 5, 2, np.nan])})

    def key(col):
        if col.name == 'a':
            return -col
        else:
            return col
    result = df.sort_values(by='a', key=key)
    expected = df.iloc[[1, 3, 4, 0, 2, 5]]
    tm.assert_frame_equal(result, expected)
    result = df.sort_values(by=['a'], key=key)
    expected = df.iloc[[1, 3, 4, 0, 2, 5]]
    tm.assert_frame_equal(result, expected)
    result = df.sort_values(by='b', key=key)
    expected = df.iloc[[0, 1, 4, 3, 2, 5]]
    tm.assert_frame_equal(result, expected)
    result = df.sort_values(by=['a', 'b'], key=key)
    expected = df.iloc[[1, 3, 4, 0, 2, 5]]
    tm.assert_frame_equal(result, expected)