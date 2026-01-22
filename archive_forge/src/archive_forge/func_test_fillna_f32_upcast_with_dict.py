from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_f32_upcast_with_dict(self):
    ser = Series([np.nan, 1.2], dtype=np.float32)
    result = ser.fillna({0: 1})
    expected = Series([1.0, 1.2], dtype=np.float32)
    tm.assert_series_equal(result, expected)