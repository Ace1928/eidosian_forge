import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_getslice_tuple(self):
    dense = np.array([np.nan, 0, 3, 4, 0, 5, np.nan, np.nan, 0])
    sparse = SparseArray(dense)
    res = sparse[slice(4, None),]
    exp = SparseArray(dense[4:])
    tm.assert_sp_array_equal(res, exp)
    sparse = SparseArray(dense, fill_value=0)
    res = sparse[slice(4, None),]
    exp = SparseArray(dense[4:], fill_value=0)
    tm.assert_sp_array_equal(res, exp)
    msg = 'too many indices for array'
    with pytest.raises(IndexError, match=msg):
        sparse[4:, :]
    with pytest.raises(IndexError, match=msg):
        dense[4:, :]