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
def test_cmp_dt64_arraylike_tznaive(self, comparison_op):
    op = comparison_op
    dti = pd.date_range('2016-01-1', freq='MS', periods=9, tz=None)
    arr = dti._data
    assert arr.freq == dti.freq
    assert arr.tz == dti.tz
    right = dti
    expected = np.ones(len(arr), dtype=bool)
    if comparison_op.__name__ in ['ne', 'gt', 'lt']:
        expected = ~expected
    result = op(arr, arr)
    tm.assert_numpy_array_equal(result, expected)
    for other in [right, np.array(right), list(right), tuple(right), right.astype(object)]:
        result = op(arr, other)
        tm.assert_numpy_array_equal(result, expected)
        result = op(other, arr)
        tm.assert_numpy_array_equal(result, expected)