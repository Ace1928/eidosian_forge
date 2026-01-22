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
def test_wishart_invwishart_2D_rvs(self):
    dim = 3
    df = 10
    scale = np.eye(dim)
    scale[0, 1] = 0.5
    scale[1, 0] = 0.5
    w = wishart(df, scale)
    iw = invwishart(df, scale)
    np.random.seed(248042)
    w_rvs = wishart.rvs(df, scale)
    np.random.seed(248042)
    frozen_w_rvs = w.rvs()
    np.random.seed(248042)
    iw_rvs = invwishart.rvs(df, scale)
    np.random.seed(248042)
    frozen_iw_rvs = iw.rvs()
    np.random.seed(248042)
    covariances = np.random.normal(size=3)
    variances = np.r_[np.random.chisquare(df), np.random.chisquare(df - 1), np.random.chisquare(df - 2)] ** 0.5
    A = np.diag(variances)
    A[np.tril_indices(dim, k=-1)] = covariances
    D = np.linalg.cholesky(scale)
    DA = D.dot(A)
    manual_w_rvs = np.dot(DA, DA.T)
    iD = np.linalg.cholesky(np.linalg.inv(scale))
    iDA = iD.dot(A)
    manual_iw_rvs = np.linalg.inv(np.dot(iDA, iDA.T))
    assert_allclose(w_rvs, manual_w_rvs)
    assert_allclose(frozen_w_rvs, manual_w_rvs)
    assert_allclose(iw_rvs, manual_iw_rvs)
    assert_allclose(frozen_iw_rvs, manual_iw_rvs)