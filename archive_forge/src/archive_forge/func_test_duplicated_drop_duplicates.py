from itertools import product
import numpy as np
import pytest
from pandas._libs import (
from pandas import (
import pandas._testing as tm
def test_duplicated_drop_duplicates():
    idx = MultiIndex.from_arrays(([1, 2, 3, 1, 2, 3], [1, 1, 1, 1, 2, 2]))
    expected = np.array([False, False, False, True, False, False], dtype=bool)
    duplicated = idx.duplicated()
    tm.assert_numpy_array_equal(duplicated, expected)
    assert duplicated.dtype == bool
    expected = MultiIndex.from_arrays(([1, 2, 3, 2, 3], [1, 1, 1, 2, 2]))
    tm.assert_index_equal(idx.drop_duplicates(), expected)
    expected = np.array([True, False, False, False, False, False])
    duplicated = idx.duplicated(keep='last')
    tm.assert_numpy_array_equal(duplicated, expected)
    assert duplicated.dtype == bool
    expected = MultiIndex.from_arrays(([2, 3, 1, 2, 3], [1, 1, 1, 2, 2]))
    tm.assert_index_equal(idx.drop_duplicates(keep='last'), expected)
    expected = np.array([True, False, False, True, False, False])
    duplicated = idx.duplicated(keep=False)
    tm.assert_numpy_array_equal(duplicated, expected)
    assert duplicated.dtype == bool
    expected = MultiIndex.from_arrays(([2, 3, 2, 3], [1, 1, 2, 2]))
    tm.assert_index_equal(idx.drop_duplicates(keep=False), expected)