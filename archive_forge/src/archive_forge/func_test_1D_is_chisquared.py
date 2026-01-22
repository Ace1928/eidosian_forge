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
def test_1D_is_chisquared(self):
    np.random.seed(482974)
    sn = 500
    dim = 1
    scale = np.eye(dim)
    df_range = np.arange(1, 10, 2, dtype=float)
    X = np.linspace(0.1, 10, num=10)
    for df in df_range:
        w = wishart(df, scale)
        c = chi2(df)
        assert_allclose(w.var(), c.var())
        assert_allclose(w.mean(), c.mean())
        assert_allclose(w.entropy(), c.entropy())
        assert_allclose(w.pdf(X), c.pdf(X))
        rvs = w.rvs(size=sn)
        args = (df,)
        alpha = 0.01
        check_distribution_rvs('chi2', args, alpha, rvs)