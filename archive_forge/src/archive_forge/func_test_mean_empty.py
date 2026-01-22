import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import NaT
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
@pytest.mark.parametrize('skipna', [True, False])
def test_mean_empty(self, arr1d, skipna):
    arr = arr1d[:0]
    assert arr.mean(skipna=skipna) is NaT
    arr2d = arr.reshape(0, 3)
    result = arr2d.mean(axis=0, skipna=skipna)
    expected = DatetimeArray._from_sequence([NaT, NaT, NaT], dtype=arr.dtype)
    tm.assert_datetime_array_equal(result, expected)
    result = arr2d.mean(axis=1, skipna=skipna)
    expected = arr
    tm.assert_datetime_array_equal(result, expected)
    result = arr2d.mean(axis=None, skipna=skipna)
    assert result is NaT