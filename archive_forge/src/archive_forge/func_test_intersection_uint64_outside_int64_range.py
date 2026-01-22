from datetime import (
import numpy as np
import pytest
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_intersection_uint64_outside_int64_range(self, index_large):
    other = Index([2 ** 63, 2 ** 63 + 5, 2 ** 63 + 10, 2 ** 63 + 15, 2 ** 63 + 20])
    result = index_large.intersection(other)
    expected = Index(np.sort(np.intersect1d(index_large.values, other.values)))
    tm.assert_index_equal(result, expected)
    result = other.intersection(index_large)
    expected = Index(np.sort(np.asarray(np.intersect1d(index_large.values, other.values))))
    tm.assert_index_equal(result, expected)