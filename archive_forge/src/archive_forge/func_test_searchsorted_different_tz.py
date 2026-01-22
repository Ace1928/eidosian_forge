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
@pytest.mark.parametrize('index', [True, False])
def test_searchsorted_different_tz(self, index):
    data = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
    arr = pd.DatetimeIndex(data, freq='D')._data.tz_localize('Asia/Tokyo')
    if index:
        arr = pd.Index(arr)
    expected = arr.searchsorted(arr[2])
    result = arr.searchsorted(arr[2].tz_convert('UTC'))
    assert result == expected
    expected = arr.searchsorted(arr[2:6])
    result = arr.searchsorted(arr[2:6].tz_convert('UTC'))
    tm.assert_equal(result, expected)