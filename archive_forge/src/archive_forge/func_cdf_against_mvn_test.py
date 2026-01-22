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
def cdf_against_mvn_test(self, dim, singular=False):
    rng = np.random.default_rng(413722918996573)
    n = 3
    w = 10 ** rng.uniform(-2, 1, size=dim)
    cov = _random_covariance(dim, w, rng, singular)
    mean = 10 ** rng.uniform(-1, 2, size=dim) * np.sign(rng.normal(size=dim))
    a = -10 ** rng.uniform(-1, 2, size=(n, dim)) + mean
    b = 10 ** rng.uniform(-1, 2, size=(n, dim)) + mean
    res = stats.multivariate_t.cdf(b, mean, cov, df=10000, lower_limit=a, allow_singular=True, random_state=rng)
    ref = stats.multivariate_normal.cdf(b, mean, cov, allow_singular=True, lower_limit=a)
    assert_allclose(res, ref, atol=0.0005)