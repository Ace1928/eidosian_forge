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
@pytest.mark.parametrize('dim', [2, 5, 8])
def test_uniform(self, dim):
    rng = np.random.default_rng(1036978481269651776)
    spherical_dist = uniform_direction(dim, seed=rng)
    v1, v2 = spherical_dist.rvs(size=2)
    v2 -= v1 @ v2 * v1
    v2 /= np.linalg.norm(v2)
    assert_allclose(v1 @ v2, 0, atol=1e-14)
    samples = spherical_dist.rvs(size=10000)
    s1 = samples @ v1
    s2 = samples @ v2
    angles = np.arctan2(s1, s2)
    angles += np.pi
    angles /= 2 * np.pi
    uniform_dist = uniform()
    kstest_result = kstest(angles, uniform_dist.cdf)
    assert kstest_result.pvalue > 0.05