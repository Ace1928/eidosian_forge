from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_axis(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((2, 2)))
    mask = DataFrame([[False, False], [False, False]])
    ser = Series([0, 1])
    expected = DataFrame([[0, 0], [1, 1]], dtype='float64')
    result = df.where(mask, ser, axis='index')
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    return_value = result.where(mask, ser, axis='index', inplace=True)
    assert return_value is None
    tm.assert_frame_equal(result, expected)
    expected = DataFrame([[0, 1], [0, 1]], dtype='float64')
    result = df.where(mask, ser, axis='columns')
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    return_value = result.where(mask, ser, axis='columns', inplace=True)
    assert return_value is None
    tm.assert_frame_equal(result, expected)