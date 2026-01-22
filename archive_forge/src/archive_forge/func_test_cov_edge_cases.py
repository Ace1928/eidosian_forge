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
def test_cov_edge_cases(self):
    cov0 = multivariate_hypergeom.cov(m=[1, 0, 0], n=1)
    cov1 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert_allclose(cov0, cov1, rtol=1e-17)
    cov3 = multivariate_hypergeom.cov(m=[0, 0, 0], n=0)
    cov4 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert_equal(cov3, cov4)
    cov5 = multivariate_hypergeom.cov(m=np.array([], dtype=int), n=0)
    cov6 = np.array([], dtype=np.float64).reshape(0, 0)
    assert_allclose(cov5, cov6, rtol=1e-17)
    assert_(cov5.shape == (0, 0))