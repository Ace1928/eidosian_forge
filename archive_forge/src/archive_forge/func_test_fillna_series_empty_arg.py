import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_fillna_series_empty_arg(using_copy_on_write):
    ser = Series([1, np.nan, 2])
    ser_orig = ser.copy()
    result = ser.fillna({})
    if using_copy_on_write:
        assert np.shares_memory(get_array(ser), get_array(result))
    else:
        assert not np.shares_memory(get_array(ser), get_array(result))
    ser.iloc[0] = 100.5
    tm.assert_series_equal(ser_orig, result)