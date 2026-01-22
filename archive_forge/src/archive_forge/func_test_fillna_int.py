from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_int(self):
    ser = Series(np.random.default_rng(2).integers(-100, 100, 50))
    return_value = ser.fillna(method='ffill', inplace=True)
    assert return_value is None
    tm.assert_series_equal(ser.fillna(method='ffill', inplace=False), ser)