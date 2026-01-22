import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ascending', [[True, False], [False, True]])
def test_sort_index_multi_already_monotonic(self, ascending):
    mi = MultiIndex.from_product([[1, 2], [3, 4]])
    ser = Series(range(len(mi)), index=mi)
    result = ser.sort_index(ascending=ascending)
    if ascending == [True, False]:
        expected = ser.take([1, 0, 3, 2])
    elif ascending == [False, True]:
        expected = ser.take([2, 3, 0, 1])
    tm.assert_series_equal(result, expected)