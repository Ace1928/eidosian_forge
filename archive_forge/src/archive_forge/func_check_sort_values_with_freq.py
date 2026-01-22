import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def check_sort_values_with_freq(self, idx):
    ordered = idx.sort_values()
    tm.assert_index_equal(ordered, idx)
    check_freq_ascending(ordered, idx, True)
    ordered = idx.sort_values(ascending=False)
    expected = idx[::-1]
    tm.assert_index_equal(ordered, expected)
    check_freq_ascending(ordered, idx, False)
    ordered, indexer = idx.sort_values(return_indexer=True)
    tm.assert_index_equal(ordered, idx)
    tm.assert_numpy_array_equal(indexer, np.array([0, 1, 2], dtype=np.intp))
    check_freq_ascending(ordered, idx, True)
    ordered, indexer = idx.sort_values(return_indexer=True, ascending=False)
    expected = idx[::-1]
    tm.assert_index_equal(ordered, expected)
    tm.assert_numpy_array_equal(indexer, np.array([2, 1, 0], dtype=np.intp))
    check_freq_ascending(ordered, idx, False)