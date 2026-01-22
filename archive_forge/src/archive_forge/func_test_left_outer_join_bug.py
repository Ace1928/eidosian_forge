import numpy as np
import pytest
from pandas._libs import join as libjoin
from pandas._libs.join import (
import pandas._testing as tm
def test_left_outer_join_bug():
    left = np.array([0, 1, 0, 1, 1, 2, 3, 1, 0, 2, 1, 2, 0, 1, 1, 2, 3, 2, 3, 2, 1, 1, 3, 0, 3, 2, 3, 0, 0, 2, 3, 2, 0, 3, 1, 3, 0, 1, 3, 0, 0, 1, 0, 3, 1, 0, 1, 0, 1, 1, 0, 2, 2, 2, 2, 2, 0, 3, 1, 2, 0, 0, 3, 1, 3, 2, 2, 0, 1, 3, 0, 2, 3, 2, 3, 3, 2, 3, 3, 1, 3, 2, 0, 0, 3, 1, 1, 1, 0, 2, 3, 3, 1, 2, 0, 3, 1, 2, 0, 2], dtype=np.intp)
    right = np.array([3, 1], dtype=np.intp)
    max_groups = 4
    lidx, ridx = libjoin.left_outer_join(left, right, max_groups, sort=False)
    exp_lidx = np.arange(len(left), dtype=np.intp)
    exp_ridx = -np.ones(len(left), dtype=np.intp)
    exp_ridx[left == 1] = 1
    exp_ridx[left == 3] = 0
    tm.assert_numpy_array_equal(lidx, exp_lidx)
    tm.assert_numpy_array_equal(ridx, exp_ridx)