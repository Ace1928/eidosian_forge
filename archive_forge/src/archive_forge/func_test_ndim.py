import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num
from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache
from scipy import stats, special as sc
from scipy.optimize._zeros_py import (_ECONVERGED, _ESIGNERR, _ECONVERR,  # noqa: F401
def test_ndim(self):
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 2, 8)
    z = np.linspace(0, 3, 13)
    wx = np.ones_like(x) * (x[1] - x[0])
    wx[0] /= 2
    wx[-1] /= 2
    wy = np.ones_like(y) * (y[1] - y[0])
    wy[0] /= 2
    wy[-1] /= 2
    wz = np.ones_like(z) * (z[1] - z[0])
    wz[0] /= 2
    wz[-1] /= 2
    q = x[:, None, None] + y[None, :, None] + z[None, None, :]
    qx = (q * wx[:, None, None]).sum(axis=0)
    qy = (q * wy[None, :, None]).sum(axis=1)
    qz = (q * wz[None, None, :]).sum(axis=2)
    r = trapezoid(q, x=x[:, None, None], axis=0)
    assert_allclose(r, qx)
    r = trapezoid(q, x=y[None, :, None], axis=1)
    assert_allclose(r, qy)
    r = trapezoid(q, x=z[None, None, :], axis=2)
    assert_allclose(r, qz)
    r = trapezoid(q, x=x, axis=0)
    assert_allclose(r, qx)
    r = trapezoid(q, x=y, axis=1)
    assert_allclose(r, qy)
    r = trapezoid(q, x=z, axis=2)
    assert_allclose(r, qz)