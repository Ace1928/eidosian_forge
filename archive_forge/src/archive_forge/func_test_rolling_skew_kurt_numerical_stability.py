from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('method', ['skew', 'kurt'])
def test_rolling_skew_kurt_numerical_stability(method):
    ser = Series(np.random.default_rng(2).random(10))
    ser_copy = ser.copy()
    expected = getattr(ser.rolling(3), method)()
    tm.assert_series_equal(ser, ser_copy)
    ser = ser + 50000
    result = getattr(ser.rolling(3), method)()
    tm.assert_series_equal(result, expected)