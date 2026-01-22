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
@pytest.mark.parametrize('dim, kappa, reference', [(3, 0.0001, 2.531024245302624), (3, 100, -1.7672931195787458), (5, 5000, -11.359032310024453), (8, 1, 3.4189526482545527)])
def test_entropy_accuracy(self, dim, kappa, reference):
    mu = np.full((dim,), 1 / np.sqrt(dim))
    entropy = vonmises_fisher(mu, kappa).entropy()
    assert_allclose(entropy, reference, rtol=2e-14)