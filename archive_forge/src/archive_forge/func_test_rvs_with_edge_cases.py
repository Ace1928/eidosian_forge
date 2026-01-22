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
@pytest.mark.parametrize('method', (None, 'boyett', 'patefield'))
@pytest.mark.parametrize('col', ([], [0]))
@pytest.mark.parametrize('row', ([], [0]))
def test_rvs_with_edge_cases(self, method, row, col):
    d = random_table(row, col)
    rv = d.rvs(10, method=method, random_state=self.get_rng())
    expected = np.zeros((10, len(row), len(col)))
    assert_equal(rv, expected)