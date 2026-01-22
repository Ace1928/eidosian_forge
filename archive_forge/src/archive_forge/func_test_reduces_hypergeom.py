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
def test_reduces_hypergeom(self):
    val1 = multivariate_hypergeom.pmf(x=[3, 1], m=[10, 5], n=4)
    val2 = hypergeom.pmf(k=3, M=15, n=4, N=10)
    assert_allclose(val1, val2, rtol=1e-08)
    val1 = multivariate_hypergeom.pmf(x=[7, 3], m=[15, 10], n=10)
    val2 = hypergeom.pmf(k=7, M=25, n=10, N=15)
    assert_allclose(val1, val2, rtol=1e-08)