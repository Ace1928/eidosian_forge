import numpy as np
import pytest
from pandas._libs import join as libjoin
from pandas._libs.join import (
import pandas._testing as tm
def test_cython_right_outer_join(self):
    left = np.array([0, 1, 2, 1, 2, 0, 0, 1, 2, 3, 3], dtype=np.intp)
    right = np.array([1, 1, 0, 4, 2, 2, 1], dtype=np.intp)
    max_group = 5
    rs, ls = left_outer_join(right, left, max_group)
    exp_ls = left.argsort(kind='mergesort')
    exp_rs = right.argsort(kind='mergesort')
    exp_li = np.array([0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 6, 7, 8, 6, 7, 8, -1])
    exp_ri = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6])
    exp_ls = exp_ls.take(exp_li)
    exp_ls[exp_li == -1] = -1
    exp_rs = exp_rs.take(exp_ri)
    exp_rs[exp_ri == -1] = -1
    tm.assert_numpy_array_equal(ls, exp_ls)
    tm.assert_numpy_array_equal(rs, exp_rs)