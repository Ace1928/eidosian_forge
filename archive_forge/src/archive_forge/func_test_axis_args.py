import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
def test_axis_args(self):
    c = np.array([[[-1, 2.5, 3, 3.5]], [[1, 6, 6, 6.5]]])
    b = np.array([[0, 0, 1, 1], [1, 1, 0, 0], [1, -1, 0, 0]])
    x = solve_circulant(c, b, baxis=1)
    assert_equal(x.shape, (4, 2, 3))
    expected = np.empty_like(x)
    expected[:, 0, :] = solve(circulant(c[0]), b.T)
    expected[:, 1, :] = solve(circulant(c[1]), b.T)
    assert_allclose(x, expected)
    x = solve_circulant(c, b, baxis=1, outaxis=-1)
    assert_equal(x.shape, (2, 3, 4))
    assert_allclose(np.moveaxis(x, -1, 0), expected)
    x = solve_circulant(np.swapaxes(c, 1, 2), b.T, caxis=1)
    assert_equal(x.shape, (4, 2, 3))
    assert_allclose(x, expected)