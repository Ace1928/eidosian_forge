import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import Timestamp
def test_quantile_empty_float64(self):
    ser = Series([], dtype='float64')
    res = ser.quantile(0.5)
    assert np.isnan(res)
    res = ser.quantile([0.5])
    exp = Series([np.nan], index=[0.5])
    tm.assert_series_equal(res, exp)