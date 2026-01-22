from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
@pytest.mark.parametrize('axis', [-2, -1, None])
def test_apply_gufunc_axis_keepdims(axis):

    def mymedian(x):
        return np.median(x, axis=-1)
    a = np.random.default_rng().standard_normal((10, 5))
    da_ = da.from_array(a, chunks=2)
    m = np.median(a, axis=-1 if not axis else axis, keepdims=True)
    dm = apply_gufunc(mymedian, '(i)->()', da_, axis=axis, keepdims=True, allow_rechunk=True)
    assert_eq(m, dm)