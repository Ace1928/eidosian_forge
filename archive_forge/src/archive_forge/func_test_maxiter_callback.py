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
def test_maxiter_callback(self):
    a, b = (-np.inf, np.inf)

    def f(x):
        return np.exp(-x * x)
    minlevel, maxlevel = (0, 2)
    maxiter = maxlevel - minlevel + 1
    kwargs = dict(minlevel=minlevel, maxlevel=maxlevel, rtol=1e-15)
    res = _tanhsinh(f, a, b, **kwargs)
    assert not res.success
    assert res.maxlevel == maxlevel

    def callback(res):
        callback.iter += 1
        callback.res = res
        assert hasattr(res, 'integral')
        assert res.status == 1
        if callback.iter == maxiter:
            raise StopIteration
    callback.iter = -1
    callback.res = None
    del kwargs['maxlevel']
    res2 = _tanhsinh(f, a, b, **kwargs, callback=callback)
    for key in res.keys():
        if key == 'status':
            assert callback.res[key] == 1
            assert res[key] == -2
            assert res2[key] == -4
        else:
            assert res2[key] == callback.res[key] == res[key]