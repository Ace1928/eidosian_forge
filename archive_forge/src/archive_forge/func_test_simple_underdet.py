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
def test_simple_underdet(self):
    for dtype in REAL_DTYPES:
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        b = np.array([1, 2], dtype=dtype)
        for lapack_driver in TestLstsq.lapack_drivers:
            for overwrite in (True, False):
                a1 = a.copy()
                b1 = b.copy()
                out = lstsq(a1, b1, lapack_driver=lapack_driver, overwrite_a=overwrite, overwrite_b=overwrite)
                x = out[0]
                r = out[2]
                assert_(r == 2, 'expected efficient rank 2, got %s' % r)
                assert_allclose(x, (-0.055555555555555, 0.111111111111111, 0.277777777777777), rtol=25 * _eps_cast(a1.dtype), atol=25 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)