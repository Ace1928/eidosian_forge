import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
@pytest.mark.xfail(_glibc_older_than('2.18'), reason='Older glibc versions are imprecise (maybe passes with SIMD?)')
@pytest.mark.xfail(IS_MUSL, reason='gh23049')
@pytest.mark.xfail(IS_WASM, reason="doesn't work")
@pytest.mark.parametrize('dtype', [np.complex64, np.complex_, np.longcomplex])
def test_loss_of_precision(self, dtype):
    """Check loss of precision in complex arc* functions"""
    info = np.finfo(dtype)
    real_dtype = dtype(0.0).real.dtype
    eps = info.eps

    def check(x, rtol):
        x = x.astype(real_dtype)
        z = x.astype(dtype)
        d = np.absolute(np.arcsinh(x) / np.arcsinh(z).real - 1)
        assert_(np.all(d < rtol), (np.argmax(d), x[np.argmax(d)], d.max(), 'arcsinh'))
        z = (1j * x).astype(dtype)
        d = np.absolute(np.arcsinh(x) / np.arcsin(z).imag - 1)
        assert_(np.all(d < rtol), (np.argmax(d), x[np.argmax(d)], d.max(), 'arcsin'))
        z = x.astype(dtype)
        d = np.absolute(np.arctanh(x) / np.arctanh(z).real - 1)
        assert_(np.all(d < rtol), (np.argmax(d), x[np.argmax(d)], d.max(), 'arctanh'))
        z = (1j * x).astype(dtype)
        d = np.absolute(np.arctanh(x) / np.arctan(z).imag - 1)
        assert_(np.all(d < rtol), (np.argmax(d), x[np.argmax(d)], d.max(), 'arctan'))
    x_series = np.logspace(-20, -3.001, 200)
    x_basic = np.logspace(-2.999, 0, 10, endpoint=False)
    if dtype is np.longcomplex:
        if bad_arcsinh():
            pytest.skip('Trig functions of np.longcomplex values known to be inaccurate on aarch64 and PPC for some compilation configurations.')
        check(x_series, 50.0 * eps)
    else:
        check(x_series, 2.1 * eps)
    check(x_basic, 2.0 * eps / 0.001)
    z = np.array([1e-05 * (1 + 1j)], dtype=dtype)
    p = 9.999999999333333e-06 + 1.0000000000666667e-05j
    d = np.absolute(1 - np.arctanh(z) / p)
    assert_(np.all(d < 1e-15))
    p = 1.0000000000333334e-05 + 9.999999999666666e-06j
    d = np.absolute(1 - np.arcsinh(z) / p)
    assert_(np.all(d < 1e-15))
    p = 9.999999999333333e-06j + 1.0000000000666667e-05
    d = np.absolute(1 - np.arctan(z) / p)
    assert_(np.all(d < 1e-15))
    p = 1.0000000000333334e-05j + 9.999999999666666e-06
    d = np.absolute(1 - np.arcsin(z) / p)
    assert_(np.all(d < 1e-15))

    def check(func, z0, d=1):
        z0 = np.asarray(z0, dtype=dtype)
        zp = z0 + abs(z0) * d * eps * 2
        zm = z0 - abs(z0) * d * eps * 2
        assert_(np.all(zp != zm), (zp, zm))
        good = abs(func(zp) - func(zm)) < 2 * eps
        assert_(np.all(good), (func, z0[~good]))
    for func in (np.arcsinh, np.arcsinh, np.arcsin, np.arctanh, np.arctan):
        pts = [rp + 1j * ip for rp in (-0.001, 0, 0.001) for ip in (-0.001, 0, 0.001) if rp != 0 or ip != 0]
        check(func, pts, 1)
        check(func, pts, 1j)
        check(func, pts, 1 + 1j)