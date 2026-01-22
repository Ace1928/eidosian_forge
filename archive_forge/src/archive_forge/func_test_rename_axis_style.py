from collections import ChainMap
import inspect
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_axis_style(self):
    df = DataFrame({'A': [1, 2], 'B': [1, 2]}, index=['X', 'Y'])
    expected = DataFrame({'a': [1, 2], 'b': [1, 2]}, index=['X', 'Y'])
    result = df.rename(str.lower, axis=1)
    tm.assert_frame_equal(result, expected)
    result = df.rename(str.lower, axis='columns')
    tm.assert_frame_equal(result, expected)
    result = df.rename({'A': 'a', 'B': 'b'}, axis=1)
    tm.assert_frame_equal(result, expected)
    result = df.rename({'A': 'a', 'B': 'b'}, axis='columns')
    tm.assert_frame_equal(result, expected)
    expected = DataFrame({'A': [1, 2], 'B': [1, 2]}, index=['x', 'y'])
    result = df.rename(str.lower, axis=0)
    tm.assert_frame_equal(result, expected)
    result = df.rename(str.lower, axis='index')
    tm.assert_frame_equal(result, expected)
    result = df.rename({'X': 'x', 'Y': 'y'}, axis=0)
    tm.assert_frame_equal(result, expected)
    result = df.rename({'X': 'x', 'Y': 'y'}, axis='index')
    tm.assert_frame_equal(result, expected)
    result = df.rename(mapper=str.lower, axis='index')
    tm.assert_frame_equal(result, expected)