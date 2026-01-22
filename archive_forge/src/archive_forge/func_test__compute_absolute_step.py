import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test__compute_absolute_step():
    methods = ['2-point', '3-point', 'cs']
    x0 = np.array([1e-05, 0, 1, 100000.0])
    EPS = np.finfo(np.float64).eps
    relative_step = {'2-point': EPS ** 0.5, '3-point': EPS ** (1 / 3), 'cs': EPS ** 0.5}
    f0 = np.array(1.0)
    for method in methods:
        rel_step = relative_step[method]
        correct_step = np.array([rel_step, rel_step * 1.0, rel_step * 1.0, rel_step * np.abs(x0[3])])
        abs_step = _compute_absolute_step(None, x0, f0, method)
        assert_allclose(abs_step, correct_step)
        sign_x0 = (-x0 >= 0).astype(float) * 2 - 1
        abs_step = _compute_absolute_step(None, -x0, f0, method)
        assert_allclose(abs_step, sign_x0 * correct_step)
    rel_step = np.array([0.1, 1, 10, 100])
    correct_step = np.array([rel_step[0] * x0[0], relative_step['2-point'], rel_step[2] * 1.0, rel_step[3] * np.abs(x0[3])])
    abs_step = _compute_absolute_step(rel_step, x0, f0, '2-point')
    assert_allclose(abs_step, correct_step)
    sign_x0 = (-x0 >= 0).astype(float) * 2 - 1
    abs_step = _compute_absolute_step(rel_step, -x0, f0, '2-point')
    assert_allclose(abs_step, sign_x0 * correct_step)