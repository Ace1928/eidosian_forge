import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_take_filling_fill_value(self):
    sparse = SparseArray([np.nan, 0, 1, 0, 4], fill_value=0)
    result = sparse.take(np.array([1, 0, -1]))
    expected = SparseArray([0, np.nan, 4], fill_value=0)
    tm.assert_sp_array_equal(result, expected)
    result = sparse.take(np.array([1, 0, -1]), allow_fill=True)
    expected = SparseArray([0, np.nan, np.nan], fill_value=0)
    tm.assert_sp_array_equal(result, expected)
    result = sparse.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
    expected = SparseArray([0, np.nan, 4], fill_value=0)
    tm.assert_sp_array_equal(result, expected)
    msg = "Invalid value in 'indices'."
    with pytest.raises(ValueError, match=msg):
        sparse.take(np.array([1, 0, -2]), allow_fill=True)
    with pytest.raises(ValueError, match=msg):
        sparse.take(np.array([1, 0, -5]), allow_fill=True)
    msg = "out of bounds value in 'indices'"
    with pytest.raises(IndexError, match=msg):
        sparse.take(np.array([1, -6]))
    with pytest.raises(IndexError, match=msg):
        sparse.take(np.array([1, 5]))
    with pytest.raises(IndexError, match=msg):
        sparse.take(np.array([1, 5]), fill_value=True)