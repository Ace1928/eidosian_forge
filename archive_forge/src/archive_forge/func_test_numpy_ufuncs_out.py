import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
from pandas.core.arrays import BooleanArray
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
def test_numpy_ufuncs_out(index):
    result = index == index
    out = np.empty(index.shape, dtype=bool)
    np.equal(index, index, out=out)
    tm.assert_numpy_array_equal(out, result)
    if not index._is_multi:
        out = np.empty(index.shape, dtype=bool)
        np.equal(index.array, index.array, out=out)
        tm.assert_numpy_array_equal(out, result)