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
def test_definition(self):

    def norm(i, e):
        return i * e / sum(e)
    np.random.seed(123)
    eigs = [norm(i, np.random.uniform(size=i)) for i in range(2, 6)]
    eigs.append([4, 0, 0, 0])
    ones = [[1.0] * len(e) for e in eigs]
    xs = [random_correlation.rvs(e) for e in eigs]
    dets = [np.fabs(np.linalg.det(x)) for x in xs]
    dets_known = [np.prod(e) for e in eigs]
    assert_allclose(dets, dets_known, rtol=1e-13, atol=1e-13)
    diags = [np.diag(x) for x in xs]
    for a, b in zip(diags, ones):
        assert_allclose(a, b, rtol=1e-13)
    for x in xs:
        assert_allclose(x, x.T, rtol=1e-13)