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
def test_scalar_a_and_1D_b(self):
    a = 1
    b = [1, 2, 3]
    x = solve(a, b)
    assert_array_almost_equal(x.ravel(), b)
    assert_(x.shape == (3,), 'Scalar_a_1D_b test returned wrong shape')