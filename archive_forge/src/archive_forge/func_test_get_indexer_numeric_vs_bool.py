import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_get_indexer_numeric_vs_bool(self):
    left = Index([1, 2, 3])
    right = Index([True, False])
    res = left.get_indexer(right)
    expected = -1 * np.ones(len(right), dtype=np.intp)
    tm.assert_numpy_array_equal(res, expected)
    res = right.get_indexer(left)
    expected = -1 * np.ones(len(left), dtype=np.intp)
    tm.assert_numpy_array_equal(res, expected)
    res = left.get_indexer_non_unique(right)[0]
    expected = -1 * np.ones(len(right), dtype=np.intp)
    tm.assert_numpy_array_equal(res, expected)
    res = right.get_indexer_non_unique(left)[0]
    expected = -1 * np.ones(len(left), dtype=np.intp)
    tm.assert_numpy_array_equal(res, expected)