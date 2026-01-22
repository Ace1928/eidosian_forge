import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_issue_11134():
    alpha, n, p = (0.95, 10, 0)
    assert_equal(binom.interval(confidence=alpha, n=n, p=p), (0, 0))