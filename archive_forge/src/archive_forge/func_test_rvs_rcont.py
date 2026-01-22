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
@pytest.mark.parametrize('v', (1, 2))
def test_rvs_rcont(self, v):
    import scipy.stats._rcont as _rcont
    row = np.array([1, 3], dtype=np.int64)
    col = np.array([2, 1, 1], dtype=np.int64)
    rvs = getattr(_rcont, f'rvs_rcont{v}')
    ntot = np.sum(row)
    result = rvs(row, col, ntot, 1, self.get_rng())
    assert result.shape == (1, len(row), len(col))
    assert np.sum(result) == ntot