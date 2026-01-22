from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_axes_input_validation_01():

    def foo(x):
        return np.mean(x, axis=-1)
    a = da.random.default_rng().normal(size=(20, 30), chunks=30)
    with pytest.raises(ValueError):
        apply_gufunc(foo, '(i)->()', a, axes=0)
    apply_gufunc(foo, '(i)->()', a, axes=[0])
    apply_gufunc(foo, '(i)->()', a, axes=[(0,)])
    apply_gufunc(foo, '(i)->()', a, axes=[0, tuple()])
    apply_gufunc(foo, '(i)->()', a, axes=[(0,), tuple()])
    with pytest.raises(ValueError):
        apply_gufunc(foo, '(i)->()', a, axes=[(0, 1)])
    with pytest.raises(ValueError):
        apply_gufunc(foo, '(i)->()', a, axes=[0, 0])