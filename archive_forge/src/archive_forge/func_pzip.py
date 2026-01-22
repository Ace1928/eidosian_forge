import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
@np.vectorize
def pzip(k, a, n):
    """Naive implementation of zipfian pmf"""
    if k < 1 or k > n:
        return 0.0
    else:
        return 1 / k ** a / Hns(n, a)