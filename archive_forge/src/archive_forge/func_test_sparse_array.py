import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.arrays import SparseArray
from pandas.tests.extension import base
def test_sparse_array(self, data_for_compare: SparseArray, comparison_op, request):
    if data_for_compare.dtype.fill_value == 0 and comparison_op.__name__ != 'gt':
        mark = pytest.mark.xfail(reason='Wrong fill_value')
        request.applymarker(mark)
    ser = pd.Series(data_for_compare)
    arr = data_for_compare + 1
    self._compare_other(ser, data_for_compare, comparison_op, arr)
    arr = data_for_compare * 2
    self._compare_other(ser, data_for_compare, comparison_op, arr)