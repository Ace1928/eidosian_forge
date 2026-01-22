import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('dtype', ['boolean', 'Int64', 'Float64'])
@pytest.mark.parametrize('data,values,expected', [([0, 1, 0], [1], [False, True, False]), ([0, 1, 0], [1, pd.NA], [False, True, False]), ([0, pd.NA, 0], [1, 0], [True, False, True]), ([0, 1, pd.NA], [1, pd.NA], [False, True, True]), ([0, 1, pd.NA], [1, np.nan], [False, True, False]), ([0, pd.NA, pd.NA], [np.nan, pd.NaT, None], [False, False, False])])
def test_isin_masked_types(self, dtype, data, values, expected):
    ser = Series(data, dtype=dtype)
    result = ser.isin(values)
    expected = Series(expected, dtype='boolean')
    tm.assert_series_equal(result, expected)