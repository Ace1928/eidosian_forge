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
def test_simple_singular(self):
    a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    a_pinv = pinv(a)
    expected = array([[-0.638888889, -0.166666667, 0.305555556], [-0.0555555556, 1.30136518e-16, 0.0555555556], [0.527777778, 0.166666667, -0.194444444]])
    assert_array_almost_equal(a_pinv, expected)