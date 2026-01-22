import numpy as np
import pytest
from pandas import (
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('func', ['min', 'max'])
@pytest.mark.parametrize('data', [np.array([]), np.array([np.nan, np.nan])])
@pytest.mark.parametrize('dtype,expected', [(SparseDtype(np.float64, np.nan), np.nan), (SparseDtype(np.float64, 5.0), np.nan), (SparseDtype('datetime64[ns]', NaT), NaT), (SparseDtype('datetime64[ns]', Timestamp('2018-05-05')), NaT)])
def test_na_value_if_no_valid_values(self, func, data, dtype, expected):
    arr = SparseArray(data, dtype=dtype)
    result = getattr(arr, func)()
    if expected is NaT:
        assert result is NaT or np.isnat(result)
    else:
        assert np.isnan(result)