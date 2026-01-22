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
def test_reproducibility(self):
    rng = np.random.RandomState(4)
    loc = rng.uniform(size=3)
    shape = np.eye(3)
    dist1 = multivariate_t(loc, shape, df=3, seed=2)
    dist2 = multivariate_t(loc, shape, df=3, seed=2)
    samples1 = dist1.rvs(size=10)
    samples2 = dist2.rvs(size=10)
    assert_equal(samples1, samples2)