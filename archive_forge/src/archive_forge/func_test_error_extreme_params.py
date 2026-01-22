import pytest
import warnings
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats
@pytest.mark.parametrize('distname, args', [('beta', (0.11, 0.11))])
def test_error_extreme_params(distname, args):
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        dist = getattr(stats, distname)(*args)
        rng = FastGeneratorInversion(dist)
    u_error, x_error = rng.evaluate_error(size=10000, random_state=980732462809709732623, x_error=True)
    if u_error >= 2.5 * 1e-10:
        assert x_error < 1e-09