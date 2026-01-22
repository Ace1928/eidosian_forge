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
@pytest.mark.parametrize('x, mu, kappa, reference', [(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0001, -2.5309242486359573), (np.array([1.0, 0.0, 0]), np.array([0.0, 0.0, 1.0]), 0.0001, -2.5310242486359575), (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 100, 2.767293119578746), (np.array([1.0, 0.0, 0]), np.array([0.0, 0.0, 1.0]), 100, -97.23270688042125), (np.array([1.0, 0.0, 0.0]), np.array([np.sqrt(0.98), np.sqrt(0.02), 0.0]), 2000, -14.337987284534103), (np.array([1.0, 0.0, 0]), np.array([1.0, 0.0, 0.0]), 2000, 5.763025393132737), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 2000, 11.526550911307156), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([np.sqrt(0.98), np.sqrt(0.02), 0.0, 0, 0.0]), 2000, -8.574461766359684), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([np.sqrt(0.8), np.sqrt(0.2), 0.0, 0, 0.0]), 2000, -199.61906708886113)])
def test_logpdf_accuracy(self, x, mu, kappa, reference):
    logpdf = vonmises_fisher(mu, kappa).logpdf(x)
    assert_allclose(logpdf, reference, rtol=1e-14)