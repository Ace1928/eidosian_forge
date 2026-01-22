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
def basic_test(self, n_points=2 ** 8, n_estimates=8, signs=np.ones(2)):
    ndim = 2
    mean = np.zeros(ndim)
    cov = np.eye(ndim)

    def func(x):
        return stats.multivariate_normal.pdf(x.T, mean, cov)
    rng = np.random.default_rng(2879434385674690281)
    qrng = stats.qmc.Sobol(ndim, seed=rng)
    a = np.zeros(ndim)
    b = np.ones(ndim) * signs
    res = qmc_quad(func, a, b, n_points=n_points, n_estimates=n_estimates, qrng=qrng)
    ref = stats.multivariate_normal.cdf(b, mean, cov, lower_limit=a)
    atol = sc.stdtrit(n_estimates - 1, 0.995) * res.standard_error
    assert_allclose(res.integral, ref, atol=atol)
    assert np.prod(signs) * res.integral > 0
    rng = np.random.default_rng(2879434385674690281)
    qrng = stats.qmc.Sobol(ndim, seed=rng)
    logres = qmc_quad(lambda *args: np.log(func(*args)), a, b, n_points=n_points, n_estimates=n_estimates, log=True, qrng=qrng)
    assert_allclose(np.exp(logres.integral), res.integral, rtol=1e-14)
    assert np.imag(logres.integral) == (np.pi if np.prod(signs) < 0 else 0)
    assert_allclose(np.exp(logres.standard_error), res.standard_error, rtol=1e-14, atol=1e-16)