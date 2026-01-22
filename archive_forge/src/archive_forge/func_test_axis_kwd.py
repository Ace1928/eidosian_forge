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
def test_axis_kwd(self):
    a = np.array([[[2, 1], [3, 4]]] * 2, 'd')
    b = norm(a, ord=np.inf, axis=(1, 0))
    c = norm(np.swapaxes(a, 0, 1), ord=np.inf, axis=(0, 1))
    d = norm(a, ord=1, axis=(0, 1))
    assert_allclose(b, c)
    assert_allclose(c, d)
    assert_allclose(b, d)
    assert_(b.shape == c.shape == d.shape)
    b = norm(a, ord=1, axis=(1, 0))
    c = norm(np.swapaxes(a, 0, 1), ord=1, axis=(0, 1))
    d = norm(a, ord=np.inf, axis=(0, 1))
    assert_allclose(b, c)
    assert_allclose(c, d)
    assert_allclose(b, d)
    assert_(b.shape == c.shape == d.shape)