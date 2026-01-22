import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_astype_dt64_to_int64(self):
    values = np.array(['NaT', '2016-01-02', '2016-01-03'], dtype='M8[ns]')
    arr = SparseArray(values)
    result = arr.astype('int64')
    expected = values.astype('int64')
    tm.assert_numpy_array_equal(result, expected)
    dtype_int64 = SparseDtype('int64', np.iinfo(np.int64).min)
    result2 = arr.astype(dtype_int64)
    tm.assert_numpy_array_equal(result2.to_numpy(), expected)
    dtype = SparseDtype('datetime64[ns]', values[1])
    arr3 = SparseArray(values, dtype=dtype)
    result3 = arr3.astype('int64')
    tm.assert_numpy_array_equal(result3, expected)