from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_setitem_str_impute_tz(self, tz_naive_fixture):
    tz = tz_naive_fixture
    data = np.array([1, 2, 3], dtype='M8[ns]')
    dtype = data.dtype if tz is None else DatetimeTZDtype(tz=tz)
    arr = DatetimeArray._from_sequence(data, dtype=dtype)
    expected = arr.copy()
    ts = pd.Timestamp('2020-09-08 16:50').tz_localize(tz)
    setter = str(ts.tz_localize(None))
    expected[0] = ts
    arr[0] = setter
    tm.assert_equal(arr, expected)
    expected[1] = ts
    arr[:2] = [setter, setter]
    tm.assert_equal(arr, expected)