import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('score', [0, 0.5, 1])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_percentile_of_score(score, axis):
    shape = (10, 20, 30)
    np.random.seed(0)
    x = np.random.rand(*shape)
    p = _resampling._percentile_of_score(x, score, axis=-1)

    def vectorized_pos(a, score, axis):
        return np.apply_along_axis(stats.percentileofscore, axis, a, score)
    p2 = vectorized_pos(x, score, axis=-1) / 100
    assert_allclose(p, p2, 1e-15)