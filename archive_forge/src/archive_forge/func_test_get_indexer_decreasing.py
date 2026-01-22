import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('stop', [0, -1, -2])
def test_get_indexer_decreasing(self, stop):
    index = RangeIndex(7, stop, -3)
    result = index.get_indexer(range(9))
    expected = np.array([-1, 2, -1, -1, 1, -1, -1, 0, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)