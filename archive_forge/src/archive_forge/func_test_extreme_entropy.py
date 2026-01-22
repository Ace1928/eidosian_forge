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
@pytest.mark.parametrize('df, dim, ref, tol', [(10, 1, 1.5212624929756808, 1e-15), (100, 1, 1.4289633653182439, 1e-13), (500, 1, 1.420939531869349, 1e-14), (1e+20, 1, 1.4189385332046727, 1e-15), (1e+100, 1, 1.4189385332046727, 1e-15), (10, 10, 15.069150450832911, 1e-15), (1000, 10, 14.19936546446673, 1e-13), (1e+20, 10, 14.189385332046728, 1e-15), (1e+100, 10, 14.189385332046728, 1e-15), (10, 100, 148.28902883192654, 1e-15), (1000, 100, 141.99155538003762, 1e-14), (1e+20, 100, 141.8938533204673, 1e-15), (1e+100, 100, 141.8938533204673, 1e-15)])
def test_extreme_entropy(self, df, dim, ref, tol):
    mvt = stats.multivariate_t(shape=np.eye(dim), df=df)
    assert_allclose(mvt.entropy(), ref, rtol=tol)