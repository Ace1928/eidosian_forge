import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import Timestamp
def test_quantile_dtype_size(self, any_int_ea_dtype):
    ser = Series([pd.NA, pd.NA, 1], dtype=any_int_ea_dtype)
    result = ser.quantile([0.1, 0.5])
    expected = Series([1, 1], dtype=any_int_ea_dtype, index=[0.1, 0.5])
    tm.assert_series_equal(result, expected)