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
def test_gh_11860(self):
    n = 88
    rng = np.random.default_rng(8879715917488330089)
    p = rng.random(n)
    p[-1] = 1e-30
    p /= np.sum(p)
    x = np.ones(n)
    logpmf = multinomial.logpmf(x, n, p)
    assert np.isfinite(logpmf)