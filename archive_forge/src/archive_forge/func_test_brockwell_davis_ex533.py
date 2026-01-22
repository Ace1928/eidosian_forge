import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.innovations import _arma_innovations, arma_innovations
from statsmodels.tsa.statespace.sarimax import SARIMAX
def test_brockwell_davis_ex533():
    nobs = 10
    ar_params = np.array([0.2])
    ma_params = np.array([0.4])
    sigma2 = 8.92
    p = len(ar_params)
    q = len(ma_params)
    m = max(p, q)
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]
    arma_process_acovf = arma_acovf(ar, ma, nobs=nobs, sigma2=sigma2)
    unconditional_variance = sigma2 * (1 + 2 * ar_params[0] * ma_params[0] + ma_params[0] ** 2) / (1 - ar_params[0] ** 2)
    assert_allclose(arma_process_acovf[0], unconditional_variance)
    arma_process_acovf /= sigma2
    unconditional_variance /= sigma2
    transformed_acovf = _arma_innovations.darma_transformed_acovf_fast(ar, ma, arma_process_acovf)
    acovf, acovf2 = (np.array(arr) for arr in transformed_acovf)
    assert_equal(acovf2.shape, (nobs - m,))
    assert_allclose(acovf2[0], 1 + ma_params[0] ** 2)
    assert_allclose(acovf2[1], ma_params[0])
    assert_allclose(acovf2[2:], 0)
    assert_equal(acovf.shape, (m * 2, m * 2))
    ix = np.diag_indices_from(acovf)
    ix_lower = (ix[0][:-1] + 1, ix[1][:-1])
    assert_allclose(acovf[ix][:m], unconditional_variance)
    assert_allclose(acovf[ix_lower][:m], ma_params[0])
    out = _arma_innovations.darma_innovations_algo_fast(nobs, ar_params, ma_params, acovf, acovf2)
    theta = np.array(out[0])
    v = np.array(out[1])
    desired_v = np.zeros(nobs)
    desired_v[0] = unconditional_variance
    for i in range(1, nobs):
        desired_v[i] = 1 + (1 - 1 / desired_v[i - 1]) * ma_params[0] ** 2
    assert_allclose(v, desired_v)
    assert_equal(theta.shape, (nobs, m + 1))
    desired_theta = np.zeros(nobs)
    for i in range(1, nobs):
        desired_theta[i] = ma_params[0] / desired_v[i - 1]
    assert_allclose(theta[:, 0], desired_theta)
    assert_allclose(theta[:, 1:], 0)
    endog = np.array([-1.1, 0.514, 0.116, -0.845, 0.872, -0.467, -0.977, -1.699, -1.228, -1.093])
    u = _arma_innovations.darma_innovations_filter(endog, ar_params, ma_params, theta)
    desired_hat = np.array([0, -0.54, 0.5068, -0.1321, -0.4539, 0.7046, -0.562, -0.3614, -0.8748, -0.3869])
    desired_u = endog - desired_hat
    assert_allclose(u, desired_u, atol=0.0001)