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
@pytest.mark.parametrize('scale,method', ((1, 'boyett'), (100, 'patefield')))
def test_process_rvs_method_on_None(self, scale, method):
    row = np.array([1, 3]) * scale
    col = np.array([2, 1, 1]) * scale
    ct = random_table
    expected = ct.rvs(row, col, method=method, random_state=1)
    got = ct.rvs(row, col, method=None, random_state=1)
    assert_equal(expected, got)