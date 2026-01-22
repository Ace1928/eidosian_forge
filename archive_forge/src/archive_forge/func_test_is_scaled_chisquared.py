import pickle
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from .test_continuous_basic import check_distribution_rvs
import numpy
import numpy as np
import scipy.linalg
from scipy.stats._multivariate import (_PSD,
from scipy.stats import (multivariate_normal, multivariate_hypergeom,
from scipy.stats import _covariance, Covariance
from scipy import stats
from scipy.integrate import romb, qmc_quad, tplquad
from scipy.special import multigammaln
from scipy._lib._pep440 import Version
from .common_tests import check_random_state_property
from .data._mvt import _qsimvtv
from unittest.mock import patch
def test_is_scaled_chisquared(self):
    np.random.seed(482974)
    sn = 500
    df = 10
    dim = 4
    scale = np.diag(np.arange(4) + 1)
    scale[np.tril_indices(4, k=-1)] = np.arange(6)
    scale = np.dot(scale.T, scale)
    lamda = np.ones((dim, 1))
    sigma_lamda = lamda.T.dot(scale).dot(lamda).squeeze()
    w = wishart(df, sigma_lamda)
    c = chi2(df, scale=sigma_lamda)
    assert_allclose(w.var(), c.var())
    assert_allclose(w.mean(), c.mean())
    assert_allclose(w.entropy(), c.entropy())
    X = np.linspace(0.1, 10, num=10)
    assert_allclose(w.pdf(X), c.pdf(X))
    rvs = w.rvs(size=sn)
    args = (df, 0, sigma_lamda)
    alpha = 0.01
    check_distribution_rvs('chi2', args, alpha, rvs)