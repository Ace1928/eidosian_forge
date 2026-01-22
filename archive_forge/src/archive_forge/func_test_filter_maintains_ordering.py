from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_filter_maintains_ordering():
    df = DataFrame({'pid': [1, 1, 1, 2, 2, 3, 3, 3], 'tag': [23, 45, 62, 24, 45, 34, 25, 62]})
    s = df['pid']
    grouped = df.groupby('tag')
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = df.iloc[[1, 2, 4, 7]]
    tm.assert_frame_equal(actual, expected)
    grouped = s.groupby(df['tag'])
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = s.iloc[[1, 2, 4, 7]]
    tm.assert_series_equal(actual, expected)
    df.index = np.arange(len(df) - 1, -1, -1)
    s = df['pid']
    grouped = df.groupby('tag')
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = df.iloc[[1, 2, 4, 7]]
    tm.assert_frame_equal(actual, expected)
    grouped = s.groupby(df['tag'])
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = s.iloc[[1, 2, 4, 7]]
    tm.assert_series_equal(actual, expected)
    SHUFFLED = [4, 6, 7, 2, 1, 0, 5, 3]
    df.index = df.index[SHUFFLED]
    s = df['pid']
    grouped = df.groupby('tag')
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = df.iloc[[1, 2, 4, 7]]
    tm.assert_frame_equal(actual, expected)
    grouped = s.groupby(df['tag'])
    actual = grouped.filter(lambda x: len(x) > 1)
    expected = s.iloc[[1, 2, 4, 7]]
    tm.assert_series_equal(actual, expected)