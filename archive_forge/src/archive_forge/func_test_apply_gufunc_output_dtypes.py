from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
@pytest.mark.parametrize('output_dtypes', [int, (int,)])
def test_apply_gufunc_output_dtypes(output_dtypes):

    def foo(x):
        return y
    x = np.random.default_rng().standard_normal(10)
    y = x.astype(int)
    dy = apply_gufunc(foo, '()->()', x, output_dtypes=output_dtypes)
    assert_eq(y, dy)