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
def test_rvs_cov(self):
    rng = self.get_rng()
    row = [2, 6]
    col = [1, 3, 4]
    rvs1 = random_table.rvs(row, col, size=10000, method='boyett', random_state=rng)
    rvs2 = random_table.rvs(row, col, size=10000, method='patefield', random_state=rng)
    cov1 = np.var(rvs1, axis=0)
    cov2 = np.var(rvs2, axis=0)
    assert_allclose(cov1, cov2, atol=0.02)