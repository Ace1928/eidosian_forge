import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
@pytest.mark.parametrize('q', [7, [1, 7]])
@pytest.mark.parametrize(argnames='axis', argvalues=[None, 1, (1,), (0, 1), (-3, -1)])
@pytest.mark.filterwarnings('ignore:All-NaN slice:RuntimeWarning')
def test_keepdims_out(self, q, axis):
    d = np.ones((3, 5, 7, 11))
    w = np.random.random((4, 200)) * np.array(d.shape)[:, None]
    w = w.astype(np.intp)
    d[tuple(w)] = np.nan
    if axis is None:
        shape_out = (1,) * d.ndim
    else:
        axis_norm = normalize_axis_tuple(axis, d.ndim)
        shape_out = tuple((1 if i in axis_norm else d.shape[i] for i in range(d.ndim)))
    shape_out = np.shape(q) + shape_out
    out = np.empty(shape_out)
    result = np.nanpercentile(d, q, axis=axis, keepdims=True, out=out)
    assert result is out
    assert_equal(result.shape, shape_out)