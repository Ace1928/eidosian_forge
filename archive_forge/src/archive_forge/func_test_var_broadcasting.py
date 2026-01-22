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
def test_var_broadcasting(self):
    var0 = multivariate_hypergeom.var(m=[10, 5], n=[4, 8])
    var1 = multivariate_hypergeom.var(m=[10, 5], n=4)
    var2 = multivariate_hypergeom.var(m=[10, 5], n=8)
    assert_allclose(var0[0], var1, rtol=1e-08)
    assert_allclose(var0[1], var2, rtol=1e-08)
    var3 = multivariate_hypergeom.var(m=[[10, 5], [10, 14]], n=[4, 8])
    var4 = [[0.6984127, 0.6984127], [1.352657, 1.352657]]
    assert_allclose(var3, var4, rtol=1e-08)
    var5 = multivariate_hypergeom.var(m=[[5], [10]], n=[5, 10])
    var6 = [[0.0], [0.0]]
    assert_allclose(var5, var6, rtol=1e-08)