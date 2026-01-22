import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
@pytest.mark.parametrize('dtypes', itertools.product(*[(int, float)] * 3))
def test_betabinom_stats_a_and_b_integers_gh18026(dtypes):
    n_type, a_type, b_type = dtypes
    n, a, b = (n_type(10), a_type(2), b_type(3))
    assert_allclose(betabinom.stats(n, a, b, moments='k'), -0.6904761904761907)