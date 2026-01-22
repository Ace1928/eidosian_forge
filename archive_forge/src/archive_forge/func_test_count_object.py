from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_count_object():
    df = DataFrame({'a': ['a'] * 3 + ['b'] * 3, 'c': [2] * 3 + [3] * 3})
    result = df.groupby('c').a.count()
    expected = Series([3, 3], index=Index([2, 3], name='c'), name='a')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'a': ['a', np.nan, np.nan] + ['b'] * 3, 'c': [2] * 3 + [3] * 3})
    result = df.groupby('c').a.count()
    expected = Series([1, 3], index=Index([2, 3], name='c'), name='a')
    tm.assert_series_equal(result, expected)