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
def test_entropy_with_covariance(self):
    _A = np.array([[1.42, 0.09, -0.49, 0.17, 0.74], [-1.13, -0.01, 0.71, 0.4, -0.56], [1.07, 0.44, -0.28, -0.44, 0.29], [-1.5, -0.94, -0.67, 0.73, -1.1], [0.17, -0.08, 1.46, -0.32, 1.36]])
    cov = _A @ _A.T
    df = 1e+20
    mul_t_entropy = stats.multivariate_t.entropy(shape=cov, df=df)
    mul_norm_entropy = multivariate_normal(None, cov=cov).entropy()
    assert_allclose(mul_t_entropy, mul_norm_entropy, rtol=1e-15)
    df1 = 765
    df2 = 768
    _entropy1 = stats.multivariate_t.entropy(shape=cov, df=df1)
    _entropy2 = stats.multivariate_t.entropy(shape=cov, df=df2)
    assert_allclose(_entropy1, _entropy2, rtol=1e-05)