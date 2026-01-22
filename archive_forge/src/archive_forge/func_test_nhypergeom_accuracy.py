import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_nhypergeom_accuracy():
    np.random.seed(0)
    x = nhypergeom.rvs(22, 7, 11, size=100)
    np.random.seed(0)
    p = np.random.uniform(size=100)
    y = nhypergeom.ppf(p, 22, 7, 11)
    assert_equal(x, y)