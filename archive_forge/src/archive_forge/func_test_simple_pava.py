import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy.optimize._pava_pybind import pava
from scipy.optimize import isotonic_regression
def test_simple_pava(self):
    y = np.array([8, 4, 8, 2, 2, 0, 8], dtype=np.float64)
    w = np.ones_like(y)
    r = np.full(shape=y.shape[0] + 1, fill_value=-1, dtype=np.intp)
    pava(y, w, r)
    assert_allclose(y, [4, 4, 4, 4, 4, 4, 8])
    assert_allclose(w, [6, 1, 1, 1, 1, 1, 1])
    assert_allclose(r, [0, 6, 7, -1, -1, -1, -1, -1])