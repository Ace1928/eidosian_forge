import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_ragged_min(self, ragged):
    df = ragged
    result = df.rolling(window='1s', min_periods=1).min()
    expected = df.copy()
    expected['B'] = [0.0, 1, 2, 3, 4]
    tm.assert_frame_equal(result, expected)
    result = df.rolling(window='2s', min_periods=1).min()
    expected = df.copy()
    expected['B'] = [0.0, 1, 1, 3, 3]
    tm.assert_frame_equal(result, expected)
    result = df.rolling(window='5s', min_periods=1).min()
    expected = df.copy()
    expected['B'] = [0.0, 0, 0, 1, 1]
    tm.assert_frame_equal(result, expected)