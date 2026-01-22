import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_constructor_float32(self):
    data = np.array([1.0, np.nan, 3], dtype=np.float32)
    arr = SparseArray(data, dtype=np.float32)
    assert arr.dtype == SparseDtype(np.float32)
    tm.assert_numpy_array_equal(arr.sp_values, np.array([1, 3], dtype=np.float32))
    tm.assert_numpy_array_equal(arr.sp_index.indices, np.array([0, 2], dtype=np.int32))
    dense = arr.to_dense()
    assert dense.dtype == np.float32
    tm.assert_numpy_array_equal(dense, data)