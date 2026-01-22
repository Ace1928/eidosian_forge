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
def test_frozen_matrix_normal(self):
    for i in range(1, 5):
        for j in range(1, 5):
            M = np.full((i, j), 0.3)
            U = 0.5 * np.identity(i) + np.full((i, i), 0.5)
            V = 0.7 * np.identity(j) + np.full((j, j), 0.3)
            frozen = matrix_normal(mean=M, rowcov=U, colcov=V)
            rvs1 = frozen.rvs(random_state=1234)
            rvs2 = matrix_normal.rvs(mean=M, rowcov=U, colcov=V, random_state=1234)
            assert_equal(rvs1, rvs2)
            X = frozen.rvs(random_state=1234)
            pdf1 = frozen.pdf(X)
            pdf2 = matrix_normal.pdf(X, mean=M, rowcov=U, colcov=V)
            assert_equal(pdf1, pdf2)
            logpdf1 = frozen.logpdf(X)
            logpdf2 = matrix_normal.logpdf(X, mean=M, rowcov=U, colcov=V)
            assert_equal(logpdf1, logpdf2)