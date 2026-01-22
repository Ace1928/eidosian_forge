from datetime import (
import numpy as np
import pytest
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('dtype', ['int64', 'uint64'])
def test_int_float_union_dtype(self, dtype):
    index = Index([0, 2, 3], dtype=dtype)
    other = Index([0.5, 1.5], dtype=np.float64)
    expected = Index([0.0, 0.5, 1.5, 2.0, 3.0], dtype=np.float64)
    result = index.union(other)
    tm.assert_index_equal(result, expected)
    result = other.union(index)
    tm.assert_index_equal(result, expected)