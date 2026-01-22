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
def test_multiple_entry_calls(self):
    np.random.seed(2846)
    n = np.random.randint(1, 32)
    alpha = np.random.uniform(1e-09, 100, n)
    d = dirichlet(alpha)
    num_tests = 10
    num_multiple = 5
    xm = None
    for i in range(num_tests):
        for m in range(num_multiple):
            x = np.random.uniform(1e-09, 100, n)
            x /= np.sum(x)
            if xm is not None:
                xm = np.vstack((xm, x))
            else:
                xm = x
        rm = d.pdf(xm.T)
        rs = None
        for xs in xm:
            r = d.pdf(xs)
            if rs is not None:
                rs = np.append(rs, r)
            else:
                rs = r
        assert_array_almost_equal(rm, rs)