import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_int_array_comparison(self, kind):
    dtype = 'int64'
    values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
    rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)
    a = SparseArray(values, dtype=dtype, kind=kind)
    b = SparseArray(rvalues, dtype=dtype, kind=kind)
    self._check_comparison_ops(a, b, values, rvalues)
    self._check_comparison_ops(a, b * 0, values, rvalues * 0)
    a = SparseArray(values, dtype=dtype, kind=kind, fill_value=0)
    b = SparseArray(rvalues, dtype=dtype, kind=kind)
    self._check_comparison_ops(a, b, values, rvalues)
    a = SparseArray(values, dtype=dtype, kind=kind, fill_value=0)
    b = SparseArray(rvalues, dtype=dtype, kind=kind, fill_value=0)
    self._check_comparison_ops(a, b, values, rvalues)
    a = SparseArray(values, dtype=dtype, kind=kind, fill_value=1)
    b = SparseArray(rvalues, dtype=dtype, kind=kind, fill_value=2)
    self._check_comparison_ops(a, b, values, rvalues)