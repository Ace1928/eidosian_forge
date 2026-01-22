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
def test_pos_and_sym(self):
    A = np.arange(1, 10).reshape(3, 3)
    x = solve(np.tril(A) / 9, np.ones(3), assume_a='pos')
    assert_array_almost_equal(x, [9.0, 1.8, 1.0])
    x = solve(np.tril(A) / 9, np.ones(3), assume_a='sym')
    assert_array_almost_equal(x, [9.0, 1.8, 1.0])