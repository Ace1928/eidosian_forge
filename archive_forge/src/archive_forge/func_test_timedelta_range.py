import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_timedelta_range(self):
    expected = to_timedelta(np.arange(5), unit='D')
    result = timedelta_range('0 days', periods=5, freq='D')
    tm.assert_index_equal(result, expected)
    expected = to_timedelta(np.arange(11), unit='D')
    result = timedelta_range('0 days', '10 days', freq='D')
    tm.assert_index_equal(result, expected)
    expected = to_timedelta(np.arange(5), unit='D') + Second(2) + Day()
    result = timedelta_range('1 days, 00:00:02', '5 days, 00:00:02', freq='D')
    tm.assert_index_equal(result, expected)
    expected = to_timedelta([1, 3, 5, 7, 9], unit='D') + Second(2)
    result = timedelta_range('1 days, 00:00:02', periods=5, freq='2D')
    tm.assert_index_equal(result, expected)
    expected = to_timedelta(np.arange(50), unit='min') * 30
    result = timedelta_range('0 days', freq='30min', periods=50)
    tm.assert_index_equal(result, expected)