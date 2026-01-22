import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_float_same_index_without_nans(self, kind, mix, all_arithmetic_functions):
    op = all_arithmetic_functions
    values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
    rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])
    a = SparseArray(values, kind=kind, fill_value=0)
    b = SparseArray(rvalues, kind=kind, fill_value=0)
    self._check_numeric_ops(a, b, values, rvalues, mix, op)