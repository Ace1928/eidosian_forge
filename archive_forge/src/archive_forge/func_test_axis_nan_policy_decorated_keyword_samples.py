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
def test_axis_nan_policy_decorated_keyword_samples():
    shape = (2, 8, 9, 10)
    rng = np.random.default_rng(0)
    x = rng.random(shape)
    x[0, 0, 0, 0] = np.nan
    res1 = stats.mannwhitneyu(*x)
    res2 = stats.mannwhitneyu(x=x[0], y=x[1])
    assert_equal(res1, res2)
    message = 'mannwhitneyu() got multiple values for argument'
    with pytest.raises(TypeError, match=re.escape(message)):
        stats.mannwhitneyu(*x, x=x[0], y=x[1])