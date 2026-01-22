import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_take_different_fill_value(self):
    sparse = pd.array([0.0], dtype=SparseDtype('float64', fill_value=0.0))
    result = sparse.take([0, -1], allow_fill=True, fill_value=np.nan)
    expected = pd.array([0, np.nan], dtype=sparse.dtype)
    tm.assert_sp_array_equal(expected, result)