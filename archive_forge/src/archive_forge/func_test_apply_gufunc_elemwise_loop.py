from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_elemwise_loop():

    def foo(x):
        assert x.shape in ((2,), (1,))
        return 2 * x
    a = da.from_array(np.array([1, 2, 3]), chunks=2, name='a')
    z = apply_gufunc(foo, '()->()', a, output_dtypes=int)
    assert z.chunks == ((2, 1),)
    assert_eq(z, np.array([2, 4, 6]))