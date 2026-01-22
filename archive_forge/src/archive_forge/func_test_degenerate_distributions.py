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
def test_degenerate_distributions(self):
    for n in range(1, 5):
        z = np.random.randn(n)
        for k in range(1, n):
            s = np.random.randn(k, k)
            cov_kk = np.dot(s, s.T)
            cov_nn = np.zeros((n, n))
            cov_nn[:k, :k] = cov_kk
            x = np.zeros(n)
            x[:k] = z[:k]
            u = _sample_orthonormal_matrix(n)
            cov_rr = np.dot(u, np.dot(cov_nn, u.T))
            y = np.dot(u, x)
            distn_kk = multivariate_normal(np.zeros(k), cov_kk, allow_singular=True)
            distn_nn = multivariate_normal(np.zeros(n), cov_nn, allow_singular=True)
            distn_rr = multivariate_normal(np.zeros(n), cov_rr, allow_singular=True)
            assert_equal(distn_kk.cov_object.rank, k)
            assert_equal(distn_nn.cov_object.rank, k)
            assert_equal(distn_rr.cov_object.rank, k)
            pdf_kk = distn_kk.pdf(x[:k])
            pdf_nn = distn_nn.pdf(x)
            pdf_rr = distn_rr.pdf(y)
            assert_allclose(pdf_kk, pdf_nn)
            assert_allclose(pdf_kk, pdf_rr)
            logpdf_kk = distn_kk.logpdf(x[:k])
            logpdf_nn = distn_nn.logpdf(x)
            logpdf_rr = distn_rr.logpdf(y)
            assert_allclose(logpdf_kk, logpdf_nn)
            assert_allclose(logpdf_kk, logpdf_rr)
            y_orth = y + u[:, -1]
            pdf_rr_orth = distn_rr.pdf(y_orth)
            logpdf_rr_orth = distn_rr.logpdf(y_orth)
            assert_equal(pdf_rr_orth, 0.0)
            assert_equal(logpdf_rr_orth, -np.inf)