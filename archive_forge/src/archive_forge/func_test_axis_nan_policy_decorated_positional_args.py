from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats import norm  # type: ignore[attr-defined]
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays
from scipy._lib._util import AxisError
def test_axis_nan_policy_decorated_positional_args():
    shape = (3, 8, 9, 10)
    rng = np.random.default_rng(0)
    x = rng.random(shape)
    x[0, 0, 0, 0] = np.nan
    stats.kruskal(*x)
    message = "kruskal() got an unexpected keyword argument 'samples'"
    with pytest.raises(TypeError, match=re.escape(message)):
        stats.kruskal(samples=x)
    with pytest.raises(TypeError, match=re.escape(message)):
        stats.kruskal(*x, samples=x)