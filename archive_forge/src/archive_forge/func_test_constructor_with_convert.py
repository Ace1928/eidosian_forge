from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_with_convert(self):
    df = DataFrame({'A': [2 ** 63 - 1]})
    result = df['A']
    expected = Series(np.asarray([2 ** 63 - 1], np.int64), name='A')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [2 ** 63]})
    result = df['A']
    expected = Series(np.asarray([2 ** 63], np.uint64), name='A')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [datetime(2005, 1, 1), True]})
    result = df['A']
    expected = Series(np.asarray([datetime(2005, 1, 1), True], np.object_), name='A')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [None, 1]})
    result = df['A']
    expected = Series(np.asarray([np.nan, 1], np.float64), name='A')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [1.0, 2]})
    result = df['A']
    expected = Series(np.asarray([1.0, 2], np.float64), name='A')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [1.0 + 2j, 3]})
    result = df['A']
    expected = Series(np.asarray([1.0 + 2j, 3], np.complex128), name='A')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [1.0 + 2j, 3.0]})
    result = df['A']
    expected = Series(np.asarray([1.0 + 2j, 3.0], np.complex128), name='A')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [1.0 + 2j, True]})
    result = df['A']
    expected = Series(np.asarray([1.0 + 2j, True], np.object_), name='A')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [1.0, None]})
    result = df['A']
    expected = Series(np.asarray([1.0, np.nan], np.float64), name='A')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [1.0 + 2j, None]})
    result = df['A']
    expected = Series(np.asarray([1.0 + 2j, np.nan], np.complex128), name='A')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [2.0, 1, True, None]})
    result = df['A']
    expected = Series(np.asarray([2.0, 1, True, None], np.object_), name='A')
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [2.0, 1, datetime(2006, 1, 1), None]})
    result = df['A']
    expected = Series(np.asarray([2.0, 1, datetime(2006, 1, 1), None], np.object_), name='A')
    tm.assert_series_equal(result, expected)