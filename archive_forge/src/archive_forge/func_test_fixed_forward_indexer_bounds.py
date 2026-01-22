import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import (
from pandas.core.indexers.objects import (
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('window_size, num_values, expected_start, expected_end', [(1, 1, [0], [1]), (1, 2, [0, 1], [1, 2]), (2, 1, [0], [1]), (2, 2, [0, 1], [2, 2]), (5, 12, range(12), list(range(5, 12)) + [12] * 5), (12, 5, range(5), [5] * 5), (0, 0, np.array([]), np.array([])), (1, 0, np.array([]), np.array([])), (0, 1, [0], [0])])
def test_fixed_forward_indexer_bounds(window_size, num_values, expected_start, expected_end, step):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)
    tm.assert_numpy_array_equal(start, np.array(expected_start[::step]), check_dtype=False)
    tm.assert_numpy_array_equal(end, np.array(expected_end[::step]), check_dtype=False)
    assert len(start) == len(end)