import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
@pytest.mark.parametrize('x, n, a, b, ref', [[5, 5000000.0, 5, 20, 1.1520944824139114e-107], [100, 50, 5, 20, 0.002855762954310226], [10000, 1000, 5, 20, 1.9648515726019154e-05]])
def test_betanbinom_pmf(self, x, n, a, b, ref):
    assert_allclose(betanbinom.pmf(x, n, a, b), ref, rtol=1e-10)