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
@pytest.mark.parametrize('scale', (1e-20, 1.0, 1e+20))
@pytest.mark.parametrize('pinv_', (pinv, pinvh))
def test_auto_rcond(scale, pinv_):
    x = np.array([[1, 0], [0, 1e-10]]) * scale
    expected = np.diag(1.0 / np.diag(x))
    x_inv = pinv_(x)
    assert_allclose(x_inv, expected)