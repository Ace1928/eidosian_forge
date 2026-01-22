import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_issue_19747():
    result = nbinom.logcdf([5, -1, 1], 5, 0.5)
    reference = [-0.47313352, -np.inf, -2.21297293]
    assert_allclose(result, reference)