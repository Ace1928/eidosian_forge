from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_filter_using_len():
    df = DataFrame({'A': np.arange(8), 'B': list('aabbbbcc'), 'C': np.arange(8)})
    grouped = df.groupby('B')
    actual = grouped.filter(lambda x: len(x) > 2)
    expected = DataFrame({'A': np.arange(2, 6), 'B': list('bbbb'), 'C': np.arange(2, 6)}, index=np.arange(2, 6, dtype=np.int64))
    tm.assert_frame_equal(actual, expected)
    actual = grouped.filter(lambda x: len(x) > 4)
    expected = df.loc[[]]
    tm.assert_frame_equal(actual, expected)
    s = df['B']
    grouped = s.groupby(s)
    actual = grouped.filter(lambda x: len(x) > 2)
    expected = Series(4 * ['b'], index=np.arange(2, 6, dtype=np.int64), name='B')
    tm.assert_series_equal(actual, expected)
    actual = grouped.filter(lambda x: len(x) > 4)
    expected = s[[]]
    tm.assert_series_equal(actual, expected)