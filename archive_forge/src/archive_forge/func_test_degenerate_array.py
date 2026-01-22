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
def test_degenerate_array(self):
    k = 10
    for n in range(2, 6):
        for r in range(1, n):
            mn = np.zeros(n)
            u = _sample_orthonormal_matrix(n)[:, :r]
            vr = np.dot(u, u.T)
            X = multivariate_normal.rvs(mean=mn, cov=vr, size=k)
            pdf = multivariate_normal.pdf(X, mean=mn, cov=vr, allow_singular=True)
            assert_equal(pdf.size, k)
            assert np.all(pdf > 0.0)
            logpdf = multivariate_normal.logpdf(X, mean=mn, cov=vr, allow_singular=True)
            assert_equal(logpdf.size, k)
            assert np.all(logpdf > -np.inf)