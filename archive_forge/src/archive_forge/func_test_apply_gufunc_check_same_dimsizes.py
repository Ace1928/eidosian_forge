from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_check_same_dimsizes():

    def foo(x, y):
        return x + y
    rng = da.random.default_rng()
    a = rng.normal(size=(3,), chunks=(2,))
    b = rng.normal(size=(4,), chunks=(2,))
    with pytest.raises(ValueError) as excinfo:
        apply_gufunc(foo, '(),()->()', a, b, output_dtypes=float, allow_rechunk=True)
    assert 'different lengths in arrays' in str(excinfo.value)