import threading
import pickle
import pytest
from copy import deepcopy
import platform
import sys
import math
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy.stats.sampling import (
from pytest import raises as assert_raises
from scipy import stats
from scipy import special
from scipy.stats import chisquare, cramervonmises
from scipy.stats._distr_params import distdiscrete, distcont
from scipy._lib._util import check_random_state
@pytest.mark.parametrize('x', x)
def test_cdf(self, x):
    dist = StandardNormal()
    rng = NumericalInversePolynomial(dist, u_resolution=1e-14)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in greater')
        sup.filter(RuntimeWarning, 'invalid value encountered in greater_equal')
        sup.filter(RuntimeWarning, 'invalid value encountered in less')
        sup.filter(RuntimeWarning, 'invalid value encountered in less_equal')
        res = rng.cdf(x)
        expected = stats.norm.cdf(x)
    assert_allclose(res, expected, rtol=1e-11, atol=1e-11)
    assert res.shape == expected.shape