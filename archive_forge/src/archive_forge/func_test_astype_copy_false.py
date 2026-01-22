import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_astype_copy_false(self):
    arr = SparseArray([1, 2, 3])
    dtype = SparseDtype(float, 0)
    result = arr.astype(dtype, copy=False)
    expected = SparseArray([1.0, 2.0, 3.0], fill_value=0.0)
    tm.assert_sp_array_equal(result, expected)