import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('dtype', [object, None])
def test_isin_dt64_values_vs_ints(self, dtype):
    dti = date_range('2013-01-01', '2013-01-05')
    ser = Series(dti)
    comps = np.asarray([1356998400000000000], dtype=dtype)
    res = dti.isin(comps)
    expected = np.array([False] * len(dti), dtype=bool)
    tm.assert_numpy_array_equal(res, expected)
    res = ser.isin(comps)
    tm.assert_series_equal(res, Series(expected))
    res = pd.core.algorithms.isin(ser, comps)
    tm.assert_numpy_array_equal(res, expected)