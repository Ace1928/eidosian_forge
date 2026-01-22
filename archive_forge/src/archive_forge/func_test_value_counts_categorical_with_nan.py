import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_value_counts_categorical_with_nan(self):
    ser = Series(['a', 'b', 'a'], dtype='category')
    exp = Series([2, 1], index=CategoricalIndex(['a', 'b']), name='count')
    res = ser.value_counts(dropna=True)
    tm.assert_series_equal(res, exp)
    res = ser.value_counts(dropna=True)
    tm.assert_series_equal(res, exp)
    series = [Series(['a', 'b', None, 'a', None, None], dtype='category'), Series(Categorical(['a', 'b', None, 'a', None, None], categories=['a', 'b']))]
    for ser in series:
        exp = Series([2, 1], index=CategoricalIndex(['a', 'b']), name='count')
        res = ser.value_counts(dropna=True)
        tm.assert_series_equal(res, exp)
        exp = Series([3, 2, 1], index=CategoricalIndex([np.nan, 'a', 'b']), name='count')
        res = ser.value_counts(dropna=False)
        tm.assert_series_equal(res, exp)
        exp = Series([2, 1, 3], index=CategoricalIndex(['a', 'b', np.nan]), name='count')
        res = ser.value_counts(dropna=False, sort=False)
        tm.assert_series_equal(res, exp)