import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_describe_one_element_ea(self):
    ser = Series([0.0], dtype='Float64')
    with tm.assert_produces_warning(None):
        result = ser.describe()
    expected = Series([1, 0, NA, 0, 0, 0, 0, 0], dtype='Float64', index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    tm.assert_series_equal(result, expected)