import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import Timestamp
def test_quantile_all_na(self, any_int_ea_dtype):
    ser = Series([pd.NA, pd.NA], dtype=any_int_ea_dtype)
    with tm.assert_produces_warning(None):
        result = ser.quantile([0.1, 0.5])
    expected = Series([pd.NA, pd.NA], dtype=any_int_ea_dtype, index=[0.1, 0.5])
    tm.assert_series_equal(result, expected)