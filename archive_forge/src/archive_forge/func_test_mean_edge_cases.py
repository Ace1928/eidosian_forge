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
def test_mean_edge_cases(self):
    mean0 = multivariate_hypergeom.mean(m=[0, 0, 0], n=0)
    assert_equal(mean0, [0.0, 0.0, 0.0])
    mean1 = multivariate_hypergeom.mean(m=[1, 0, 0], n=2)
    assert_equal(mean1, [np.nan, np.nan, np.nan])
    mean2 = multivariate_hypergeom.mean(m=[[1, 0, 0], [1, 0, 1]], n=2)
    assert_allclose(mean2, [[np.nan, np.nan, np.nan], [1.0, 0.0, 1.0]], rtol=1e-17)
    mean3 = multivariate_hypergeom.mean(m=np.array([], dtype=int), n=0)
    assert_equal(mean3, [])
    assert_(mean3.shape == (0,))