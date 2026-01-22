from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [None, 'int64', 'uint64'])
def test_intersection_mismatched_dtype(self, dtype):
    index = RangeIndex(start=0, stop=20, step=2, name='foo')
    index = Index(index, dtype=dtype)
    flt = index.astype(np.float64)
    result = index.intersection(flt)
    tm.assert_index_equal(result, index, exact=True)
    result = flt.intersection(index)
    tm.assert_index_equal(result, flt, exact=True)
    result = index.intersection(flt[1:])
    tm.assert_index_equal(result, flt[1:], exact=True)
    result = flt[1:].intersection(index)
    tm.assert_index_equal(result, flt[1:], exact=True)
    result = index.intersection(flt[:0])
    tm.assert_index_equal(result, flt[:0], exact=True)
    result = flt[:0].intersection(index)
    tm.assert_index_equal(result, flt[:0], exact=True)