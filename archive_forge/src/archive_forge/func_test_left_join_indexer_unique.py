import numpy as np
import pytest
from pandas._libs import join as libjoin
from pandas._libs.join import (
import pandas._testing as tm
@pytest.mark.parametrize('readonly', [True, False])
def test_left_join_indexer_unique(readonly):
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([2, 2, 3, 4, 4], dtype=np.int64)
    if readonly:
        a.setflags(write=False)
        b.setflags(write=False)
    result = libjoin.left_join_indexer_unique(b, a)
    expected = np.array([1, 1, 2, 3, 3], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)