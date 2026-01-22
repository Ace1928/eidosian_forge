from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_axes_args_validation():

    def add(x, y):
        return x + y
    a = da.from_array(np.array([1, 2, 3]), chunks=2, name='a')
    b = da.from_array(np.array([1, 2, 3]), chunks=2, name='b')
    with pytest.raises(ValueError):
        apply_gufunc(add, '(),()->()', a, b, 0, output_dtypes=a.dtype)