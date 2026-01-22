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
def test_simple_real(self):
    a = array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
    a = np.dot(a, a.T)
    a_pinv = pinvh(a)
    assert_array_almost_equal(np.dot(a, a_pinv), np.eye(3))