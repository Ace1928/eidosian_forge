from itertools import permutations
import numpy as np
import pytest
from pandas._libs.interval import IntervalTree
from pandas.compat import IS64
import pandas._testing as tm
@pytest.mark.parametrize('dtype, target_value, target_dtype', [('int64', 2 ** 63 + 1, 'uint64'), ('uint64', -1, 'int64')])
def test_get_indexer_non_unique_overflow(self, dtype, target_value, target_dtype):
    left, right = (np.array([0, 2], dtype=dtype), np.array([1, 3], dtype=dtype))
    tree = IntervalTree(left, right)
    target = np.array([target_value], dtype=target_dtype)
    result_indexer, result_missing = tree.get_indexer_non_unique(target)
    expected_indexer = np.array([-1], dtype='intp')
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)
    expected_missing = np.array([0], dtype='intp')
    tm.assert_numpy_array_equal(result_missing, expected_missing)