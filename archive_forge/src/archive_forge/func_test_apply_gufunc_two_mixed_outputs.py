from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_two_mixed_outputs():

    def foo():
        return (1, np.ones((2, 3), dtype=float))
    x, y = apply_gufunc(foo, '->(),(i,j)', output_dtypes=(int, float), output_sizes={'i': 2, 'j': 3})
    assert x.compute() == 1
    assert y.chunks == ((2,), (3,))
    assert_eq(y, np.ones((2, 3), dtype=float))