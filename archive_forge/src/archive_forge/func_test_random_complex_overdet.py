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
def test_random_complex_overdet(self):
    rng = np.random.RandomState(1234)
    for dtype in COMPLEX_DTYPES:
        for n, m in ((20, 15), (200, 2)):
            for lapack_driver in TestLstsq.lapack_drivers:
                for overwrite in (True, False):
                    a = np.asarray(rng.random([n, m]) + 1j * rng.random([n, m]), dtype=dtype)
                    for i in range(m):
                        a[i, i] = 20 * (0.1 + a[i, i])
                    for i in range(2):
                        b = np.asarray(rng.random([n, 3]), dtype=dtype)
                        a1 = a.copy()
                        b1 = b.copy()
                        out = lstsq(a1, b1, lapack_driver=lapack_driver, overwrite_a=overwrite, overwrite_b=overwrite)
                        x = out[0]
                        r = out[2]
                        assert_(r == m, f'expected efficient rank {m}, got {r}')
                        assert_allclose(x, direct_lstsq(a, b, cmplx=1), rtol=25 * _eps_cast(a1.dtype), atol=25 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)