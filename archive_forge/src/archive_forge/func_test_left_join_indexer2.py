import numpy as np
import pytest
from pandas._libs import join as libjoin
from pandas._libs.join import (
import pandas._testing as tm
def test_left_join_indexer2():
    idx = np.array([1, 1, 2, 5], dtype=np.int64)
    idx2 = np.array([1, 2, 5, 7, 9], dtype=np.int64)
    res, lidx, ridx = libjoin.left_join_indexer(idx2, idx)
    exp_res = np.array([1, 1, 2, 5, 7, 9], dtype=np.int64)
    tm.assert_almost_equal(res, exp_res)
    exp_lidx = np.array([0, 0, 1, 2, 3, 4], dtype=np.intp)
    tm.assert_almost_equal(lidx, exp_lidx)
    exp_ridx = np.array([0, 1, 2, 3, -1, -1], dtype=np.intp)
    tm.assert_almost_equal(ridx, exp_ridx)