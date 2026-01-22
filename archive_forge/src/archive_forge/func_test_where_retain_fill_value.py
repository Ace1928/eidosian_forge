import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_where_retain_fill_value(self):
    arr = SparseArray([np.nan, 1.0], fill_value=0)
    mask = np.array([True, False])
    res = arr._where(~mask, 1)
    exp = SparseArray([1, 1.0], fill_value=0)
    tm.assert_sp_array_equal(res, exp)
    ser = pd.Series(arr)
    res = ser.where(~mask, 1)
    tm.assert_series_equal(res, pd.Series(exp))