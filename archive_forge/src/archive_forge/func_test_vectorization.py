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
@pytest.mark.parametrize('shape', [tuple(), (12,), (3, 4), (3, 2, 2)])
def test_vectorization(self, shape):
    rng = np.random.default_rng(82456839535679456794)
    a = rng.random(shape)
    b = rng.random(shape)
    p = rng.random(shape)
    n = np.prod(shape)

    def f(x, p):
        f.ncall += 1
        f.feval += 1 if x.size == n or x.ndim <= 1 else x.shape[-1]
        return x ** p
    f.ncall = 0
    f.feval = 0

    @np.vectorize
    def _tanhsinh_single(a, b, p):
        return _tanhsinh(lambda x: x ** p, a, b)
    res = _tanhsinh(f, a, b, args=(p,))
    refs = _tanhsinh_single(a, b, p).ravel()
    attrs = ['integral', 'error', 'success', 'status', 'nfev', 'maxlevel']
    for attr in attrs:
        ref_attr = [getattr(ref, attr) for ref in refs]
        res_attr = getattr(res, attr)
        assert_allclose(res_attr.ravel(), ref_attr, rtol=1e-15)
        assert_equal(res_attr.shape, shape)
    assert np.issubdtype(res.success.dtype, np.bool_)
    assert np.issubdtype(res.status.dtype, np.integer)
    assert np.issubdtype(res.nfev.dtype, np.integer)
    assert np.issubdtype(res.maxlevel.dtype, np.integer)
    assert_equal(np.max(res.nfev), f.feval)
    assert np.max(res.maxlevel) >= 2
    assert_equal(np.max(res.maxlevel), f.ncall)