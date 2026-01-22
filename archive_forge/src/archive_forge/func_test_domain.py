import pytest
import warnings
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats
def test_domain():
    rng = FastGeneratorInversion(stats.norm(), domain=(-1, 1))
    r = rng.rvs(size=100)
    assert -1 <= r.min() < r.max() <= 1
    loc, scale = (3.5, 1.3)
    dist = stats.norm(loc=loc, scale=scale)
    rng = FastGeneratorInversion(dist, domain=(-1.5, 2))
    r = rng.rvs(size=100)
    lb, ub = (loc - scale * 1.5, loc + scale * 2)
    assert lb <= r.min() < r.max() <= ub