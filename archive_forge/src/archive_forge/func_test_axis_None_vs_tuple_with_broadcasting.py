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
def test_axis_None_vs_tuple_with_broadcasting():
    rng = np.random.default_rng(0)
    x = rng.random((5, 1))
    y = rng.random((1, 5))
    x2, y2 = np.broadcast_arrays(x, y)
    res0 = stats.mannwhitneyu(x.ravel(), y.ravel())
    res1 = stats.mannwhitneyu(x, y, axis=None)
    res2 = stats.mannwhitneyu(x, y, axis=(0, 1))
    res3 = stats.mannwhitneyu(x2.ravel(), y2.ravel())
    assert res1 == res0
    assert res2 == res0
    assert res3 != res0