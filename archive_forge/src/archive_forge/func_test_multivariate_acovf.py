import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
def test_multivariate_acovf():
    _acovf = tools._compute_multivariate_acovf_from_coefficients
    Sigma_u = np.array([[2.25, 0, 0], [0, 1.0, 0.5], [0, 0.5, 0.74]])
    Phi_1 = np.array([[0.5, 0, 0], [0.1, 0.1, 0.3], [0, 0.2, 0.3]])
    Gamma_0 = np.array([[3.0, 0.161, 0.019], [0.161, 1.172, 0.674], [0.019, 0.674, 0.954]])
    assert_allclose(_acovf([Phi_1], Sigma_u)[0], Gamma_0, atol=0.001)
    Sigma_u = np.diag([0.09, 0.04])
    Phi_1 = np.array([[0.5, 0.1], [0.4, 0.5]])
    Phi_2 = np.array([[0, 0], [0.25, 0]])
    Gamma_0 = np.array([[0.131, 0.066], [0.066, 0.181]])
    Gamma_1 = np.array([[0.072, 0.051], [0.104, 0.143]])
    Gamma_2 = np.array([[0.046, 0.04], [0.113, 0.108]])
    Gamma_3 = np.array([[0.035, 0.031], [0.093, 0.083]])
    assert_allclose(_acovf([Phi_1, Phi_2], Sigma_u, maxlag=0), [Gamma_0], atol=0.001)
    assert_allclose(_acovf([Phi_1, Phi_2], Sigma_u, maxlag=1), [Gamma_0, Gamma_1], atol=0.001)
    assert_allclose(_acovf([Phi_1, Phi_2], Sigma_u), [Gamma_0, Gamma_1], atol=0.001)
    assert_allclose(_acovf([Phi_1, Phi_2], Sigma_u, maxlag=2), [Gamma_0, Gamma_1, Gamma_2], atol=0.001)
    assert_allclose(_acovf([Phi_1, Phi_2], Sigma_u, maxlag=3), [Gamma_0, Gamma_1, Gamma_2, Gamma_3], atol=0.001)
    x = np.arange(20) * 1.0
    assert_allclose(np.squeeze(tools._compute_multivariate_sample_acovf(x, maxlag=4)), acovf(x, fft=False)[:5])