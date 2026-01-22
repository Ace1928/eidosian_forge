import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_constructor_spindex_dtype_scalar_broadcasts(self):
    arr = SparseArray(data=[1, 2], sparse_index=IntIndex(4, [1, 2]), fill_value=0, dtype=None)
    exp = SparseArray([0, 1, 2, 0], fill_value=0, dtype=None)
    tm.assert_sp_array_equal(arr, exp)
    assert arr.dtype == SparseDtype(np.int64)
    assert arr.fill_value == 0