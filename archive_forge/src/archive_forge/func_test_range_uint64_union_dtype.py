from datetime import (
import numpy as np
import pytest
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_range_uint64_union_dtype(self):
    index = RangeIndex(start=0, stop=3)
    other = Index([0, 10], dtype=np.uint64)
    result = index.union(other)
    expected = Index([0, 1, 2, 10], dtype=object)
    tm.assert_index_equal(result, expected)
    result = other.union(index)
    tm.assert_index_equal(result, expected)