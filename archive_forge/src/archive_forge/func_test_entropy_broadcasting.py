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
def test_entropy_broadcasting(self):
    ent0 = multinomial.entropy([2, 3], [0.2, 0.3])
    assert_allclose(ent0, [binom.entropy(2, 0.2), binom.entropy(3, 0.2)], rtol=1e-08)
    ent1 = multinomial.entropy([7, 8], [[0.3, 0.7], [0.4, 0.6]])
    assert_allclose(ent1, [binom.entropy(7, 0.3), binom.entropy(8, 0.4)], rtol=1e-08)
    ent2 = multinomial.entropy([[7], [8]], [[0.3, 0.7], [0.4, 0.6]])
    assert_allclose(ent2, [[binom.entropy(7, 0.3), binom.entropy(7, 0.4)], [binom.entropy(8, 0.3), binom.entropy(8, 0.4)]], rtol=1e-08)