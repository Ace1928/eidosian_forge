import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_getitem_arraylike_mask(self, arr):
    arr = SparseArray([0, 1, 2])
    result = arr[[True, False, True]]
    expected = SparseArray([0, 2])
    tm.assert_sp_array_equal(result, expected)