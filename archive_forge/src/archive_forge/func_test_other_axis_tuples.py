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
@pytest.mark.parametrize('axis', list(permutations(range(-3, 3), 2)) + [(-4, 1)])
def test_other_axis_tuples(axis):
    rng = np.random.default_rng(0)
    shape_x = (4, 5, 6)
    shape_y = (1, 6)
    x = rng.random(shape_x)
    y = rng.random(shape_y)
    axis_original = axis
    axis = tuple([i if i >= 0 else 3 + i for i in axis])
    axis = sorted(axis)
    if len(set(axis)) != len(axis):
        message = '`axis` must contain only distinct elements'
        with pytest.raises(AxisError, match=re.escape(message)):
            stats.mannwhitneyu(x, y, axis=axis_original)
        return
    if axis[0] < 0 or axis[-1] > 2:
        message = '`axis` is out of bounds for array of dimension 3'
        with pytest.raises(AxisError, match=re.escape(message)):
            stats.mannwhitneyu(x, y, axis=axis_original)
        return
    res = stats.mannwhitneyu(x, y, axis=axis_original)
    not_axis = {0, 1, 2} - set(axis)
    not_axis = next(iter(not_axis))
    x2 = x
    shape_y_broadcasted = [1, 1, 6]
    shape_y_broadcasted[not_axis] = shape_x[not_axis]
    y2 = np.broadcast_to(y, shape_y_broadcasted)
    m = x2.shape[not_axis]
    x2 = np.moveaxis(x2, axis, (1, 2))
    y2 = np.moveaxis(y2, axis, (1, 2))
    x2 = np.reshape(x2, (m, -1))
    y2 = np.reshape(y2, (m, -1))
    res2 = stats.mannwhitneyu(x2, y2, axis=1)
    np.testing.assert_array_equal(res, res2)