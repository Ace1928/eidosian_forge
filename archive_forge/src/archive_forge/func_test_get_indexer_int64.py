import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_get_indexer_int64(self):
    index = Index(range(0, 20, 2), dtype=np.int64)
    target = Index(np.arange(10), dtype=np.int64)
    indexer = index.get_indexer(target)
    expected = np.array([0, -1, 1, -1, 2, -1, 3, -1, 4, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected)
    target = Index(np.arange(10), dtype=np.int64)
    indexer = index.get_indexer(target, method='pad')
    expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected)
    target = Index(np.arange(10), dtype=np.int64)
    indexer = index.get_indexer(target, method='backfill')
    expected = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected)