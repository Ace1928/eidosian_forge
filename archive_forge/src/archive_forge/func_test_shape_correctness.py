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
def test_shape_correctness(self):
    dim = 4
    loc = np.zeros(dim)
    shape = np.eye(dim)
    df = 4.5
    x = np.zeros(dim)
    res = multivariate_t(loc, shape, df).pdf(x)
    assert np.isscalar(res)
    res = multivariate_t(loc, shape, df).logpdf(x)
    assert np.isscalar(res)
    n_samples = 7
    x = np.random.random((n_samples, dim))
    res = multivariate_t(loc, shape, df).pdf(x)
    assert res.shape == (n_samples,)
    res = multivariate_t(loc, shape, df).logpdf(x)
    assert res.shape == (n_samples,)
    res = multivariate_t(np.zeros(1), np.eye(1), 1).rvs()
    assert np.isscalar(res)
    size = 7
    res = multivariate_t(np.zeros(1), np.eye(1), 1).rvs(size=size)
    assert res.shape == (size,)