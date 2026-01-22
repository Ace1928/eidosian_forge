import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import Timestamp
def test_quantile_empty_dt64(self):
    ser = Series([], dtype='datetime64[ns]')
    res = ser.quantile(0.5)
    assert res is pd.NaT
    res = ser.quantile([0.5])
    exp = Series([pd.NaT], index=[0.5], dtype=ser.dtype)
    tm.assert_series_equal(res, exp)