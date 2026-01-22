import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
def test_gh2002_regression():
    x = np.r_[-2:2:101j]
    a = np.r_[-np.ones(50), np.ones(51)]
    expected = [stats.truncnorm.pdf(_x, _a, np.inf) for _x, _a in zip(x, a)]
    ans = stats.truncnorm.pdf(x, a, np.inf)
    npt.assert_array_almost_equal(ans, expected)