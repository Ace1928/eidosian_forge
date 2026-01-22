import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_astype_all(self, any_real_numpy_dtype):
    vals = np.array([1, 2, 3])
    arr = SparseArray(vals, fill_value=1)
    typ = np.dtype(any_real_numpy_dtype)
    res = arr.astype(typ)
    tm.assert_numpy_array_equal(res, vals.astype(any_real_numpy_dtype))