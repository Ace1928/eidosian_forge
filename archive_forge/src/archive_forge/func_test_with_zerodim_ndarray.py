import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_with_zerodim_ndarray():
    arr = SparseArray([0, 1], fill_value=0)
    result = arr * np.array(2)
    expected = arr * 2
    tm.assert_sp_array_equal(result, expected)