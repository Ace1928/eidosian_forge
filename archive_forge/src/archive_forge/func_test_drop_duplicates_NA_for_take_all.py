from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_drop_duplicates_NA_for_take_all():
    df = DataFrame({'A': [None, None, 'foo', 'bar', 'foo', 'baz', 'bar', 'qux'], 'C': [1.0, np.nan, np.nan, np.nan, 1.0, 2.0, 3, 1.0]})
    result = df.drop_duplicates('A')
    expected = df.iloc[[0, 2, 3, 5, 7]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates('A', keep='last')
    expected = df.iloc[[1, 4, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates('A', keep=False)
    expected = df.iloc[[5, 7]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates('C')
    expected = df.iloc[[0, 1, 5, 6]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates('C', keep='last')
    expected = df.iloc[[3, 5, 6, 7]]
    tm.assert_frame_equal(result, expected)
    result = df.drop_duplicates('C', keep=False)
    expected = df.iloc[[5, 6]]
    tm.assert_frame_equal(result, expected)