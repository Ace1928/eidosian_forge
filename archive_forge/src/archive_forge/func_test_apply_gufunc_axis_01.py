from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
@pytest.mark.parametrize('keepdims', [False, True])
def test_apply_gufunc_axis_01(keepdims):

    def mymedian(x):
        return np.median(x, axis=-1)
    a = np.random.default_rng().standard_normal((10, 5))
    da_ = da.from_array(a, chunks=2)
    m = np.median(a, axis=0, keepdims=keepdims)
    dm = apply_gufunc(mymedian, '(i)->()', da_, axis=0, keepdims=keepdims, allow_rechunk=True)
    assert_eq(m, dm)