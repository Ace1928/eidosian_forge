import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_her2(self):
    x = np.arange(1, 9, dtype='d').view('D')
    y = np.arange(9, 17, dtype='d').view('D')
    resxy = x[:, np.newaxis] * y.conj() + y[:, np.newaxis] * x.conj()
    resxy = np.triu(resxy)
    resxy_reverse = x[::-1, np.newaxis] * y[::-1].conj()
    resxy_reverse += y[::-1, np.newaxis] * x[::-1].conj()
    resxy_reverse = np.triu(resxy_reverse)
    u = np.c_[np.zeros(4), x, np.zeros(4)].ravel()
    v = np.c_[np.zeros(4), y, np.zeros(4)].ravel()
    for p, rtol in zip('cz', [1e-07, 1e-14]):
        f = getattr(fblas, p + 'her2', None)
        if f is None:
            continue
        assert_allclose(f(1.0, x, y), resxy, rtol=rtol)
        assert_allclose(f(1.0, x, y, n=3), resxy[:3, :3], rtol=rtol)
        assert_allclose(f(1.0, x, y, lower=True), resxy.T.conj(), rtol=rtol)
        assert_allclose(f(1.0, u, v, incx=3, offx=1, incy=3, offy=1), resxy, rtol=rtol)
        assert_allclose(f(1.0, u, v, incx=3, offx=1, incy=3, offy=1, n=3), resxy[:3, :3], rtol=rtol)
        assert_allclose(f(1.0, u, v, incx=-3, offx=1, incy=-3, offy=1), resxy_reverse, rtol=rtol)
        a = np.zeros((4, 4), 'F' if p == 'c' else 'D', 'F')
        b = f(1.0, x, y, a=a, overwrite_a=True)
        assert_allclose(a, resxy, rtol=rtol)
        b = f(2.0, x, y, a=a)
        assert_(a is not b)
        assert_allclose(b, 3 * resxy, rtol=rtol)
        assert_raises(Exception, f, 1.0, x, y, incx=0)
        assert_raises(Exception, f, 1.0, x, y, offx=5)
        assert_raises(Exception, f, 1.0, x, y, offx=-2)
        assert_raises(Exception, f, 1.0, x, y, incy=0)
        assert_raises(Exception, f, 1.0, x, y, offy=5)
        assert_raises(Exception, f, 1.0, x, y, offy=-2)
        assert_raises(Exception, f, 1.0, x, y, n=-2)
        assert_raises(Exception, f, 1.0, x, y, n=5)
        assert_raises(Exception, f, 1.0, x, y, lower=2)
        assert_raises(Exception, f, 1.0, x, y, a=np.zeros((2, 2), 'd', 'F'))