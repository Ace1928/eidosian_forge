import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def check_sort_values_without_freq(self, idx, expected):
    ordered = idx.sort_values(na_position='first')
    tm.assert_index_equal(ordered, expected)
    check_freq_nonmonotonic(ordered, idx)
    if not idx.isna().any():
        ordered = idx.sort_values()
        tm.assert_index_equal(ordered, expected)
        check_freq_nonmonotonic(ordered, idx)
    ordered = idx.sort_values(ascending=False)
    tm.assert_index_equal(ordered, expected[::-1])
    check_freq_nonmonotonic(ordered, idx)
    ordered, indexer = idx.sort_values(return_indexer=True, na_position='first')
    tm.assert_index_equal(ordered, expected)
    exp = np.array([0, 4, 3, 1, 2], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, exp)
    check_freq_nonmonotonic(ordered, idx)
    if not idx.isna().any():
        ordered, indexer = idx.sort_values(return_indexer=True)
        tm.assert_index_equal(ordered, expected)
        exp = np.array([0, 4, 3, 1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(indexer, exp)
        check_freq_nonmonotonic(ordered, idx)
    ordered, indexer = idx.sort_values(return_indexer=True, ascending=False)
    tm.assert_index_equal(ordered, expected[::-1])
    exp = np.array([2, 1, 3, 0, 4], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, exp)
    check_freq_nonmonotonic(ordered, idx)