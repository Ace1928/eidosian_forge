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
def test_mean_broadcasting(self):
    mean0 = multivariate_hypergeom.mean(m=[[3, 5], [10, 5]], n=[4, 8])
    mean1 = [[3.0 * 4.0 / 8.0, 5.0 * 4.0 / 8.0], [10.0 * 8.0 / 15.0, 5.0 * 8.0 / 15.0]]
    assert_allclose(mean0, mean1, rtol=1e-08)