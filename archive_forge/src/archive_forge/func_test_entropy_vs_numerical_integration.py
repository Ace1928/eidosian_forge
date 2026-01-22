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
@pytest.mark.parametrize('df, cov, ref, tol', [(10, np.eye(2, 2), 3.0378770664093313, 1e-14), (100, np.array([[0.5, 1], [1, 10]]), 3.55102424550609, 1e-08)])
def test_entropy_vs_numerical_integration(self, df, cov, ref, tol):
    loc = np.zeros((2,))
    mvt = stats.multivariate_t(loc, cov, df)
    assert_allclose(mvt.entropy(), ref, rtol=tol)