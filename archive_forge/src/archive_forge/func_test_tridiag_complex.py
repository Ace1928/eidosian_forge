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
def test_tridiag_complex(self):
    ab = array([[0.0, 20, 6, 2j], [1, 4, 20, 14], [-30, 1, 7, 0]])
    a = np.diag(ab[0, 1:], 1) + np.diag(ab[1, :], 0) + np.diag(ab[2, :-1], -1)
    b4 = array([10.0, 0.0, 2.0, 14j])
    b4by1 = b4.reshape(-1, 1)
    b4by2 = array([[2, 1], [-30, 4], [2, 3], [1, 3]])
    b4by4 = array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0]])
    for b in [b4, b4by1, b4by2, b4by4]:
        x = solve_banded((1, 1), ab, b)
        assert_array_almost_equal(dot(a, x), b)