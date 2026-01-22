from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_filter_and_transform_with_non_unique_string_index():
    index = list('bbbcbbab')
    df = DataFrame({'pid': [1, 1, 1, 2, 2, 3, 3, 3], 'tag': [23, 45, 62, 24, 45, 34, 25, 62]}, index=index)
    grouped_df = df.groupby('tag')
    ser = df['pid']
    grouped_ser = ser.groupby(df['tag'])
    expected_indexes = [1, 2, 4, 7]
    actual = grouped_df.filter(lambda x: len(x) > 1)
    expected = df.iloc[expected_indexes]
    tm.assert_frame_equal(actual, expected)
    actual = grouped_df.filter(lambda x: len(x) > 1, dropna=False)
    expected = df.copy().astype('float64')
    expected.iloc[[0, 3, 5, 6]] = np.nan
    tm.assert_frame_equal(actual, expected)
    actual = grouped_ser.filter(lambda x: len(x) > 1)
    expected = ser.take(expected_indexes)
    tm.assert_series_equal(actual, expected)
    actual = grouped_ser.filter(lambda x: len(x) > 1, dropna=False)
    expected = Series([np.nan, 1, 1, np.nan, 2, np.nan, np.nan, 3], index, name='pid')
    tm.assert_series_equal(actual, expected)
    actual = grouped_ser.transform(len)
    expected = Series([1, 2, 2, 1, 2, 1, 1, 2], index, name='pid')
    tm.assert_series_equal(actual, expected)
    actual = grouped_df.pid.transform(len)
    tm.assert_series_equal(actual, expected)