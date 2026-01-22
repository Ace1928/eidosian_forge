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
def test_to_corr(self):
    m = np.array([[0.1, 0], [0, 1]], dtype=float)
    m = random_correlation._to_corr(m)
    assert_allclose(m, np.array([[1, 0], [0, 0.1]]))
    with np.errstate(over='ignore'):
        g = np.array([[0, 1], [-1, 0]])
        m0 = np.array([[1e+300, 0], [0, np.nextafter(1, 0)]], dtype=float)
        m = random_correlation._to_corr(m0.copy())
        assert_allclose(m, g.T.dot(m0).dot(g))
        m0 = np.array([[0.9, 1e+300], [1e+300, 1.1]], dtype=float)
        m = random_correlation._to_corr(m0.copy())
        assert_allclose(m, g.T.dot(m0).dot(g))
    m0 = np.array([[2, 1], [1, 2]], dtype=float)
    m = random_correlation._to_corr(m0.copy())
    assert_allclose(m[0, 0], 1)
    m0 = np.array([[2 + 1e-07, 1], [1, 2]], dtype=float)
    m = random_correlation._to_corr(m0.copy())
    assert_allclose(m[0, 0], 1)