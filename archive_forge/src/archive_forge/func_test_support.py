import pytest
import warnings
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats
@pytest.mark.parametrize('distname, args, expected', [('beta', (3.5, 2.5), (0, 1)), ('norm', (), (-np.inf, np.inf))])
def test_support(distname, args, expected):
    dist = getattr(stats, distname)(*args)
    rng = FastGeneratorInversion(dist)
    assert_array_equal(rng.support(), expected)
    rng.loc = 1
    rng.scale = 2
    assert_array_equal(rng.support(), 1 + 2 * np.array(expected))