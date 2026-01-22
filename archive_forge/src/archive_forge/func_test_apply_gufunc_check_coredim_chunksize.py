from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_check_coredim_chunksize():

    def foo(x):
        return np.sum(x, axis=-1)
    a = da.random.default_rng().normal(size=(8,), chunks=3)
    with pytest.raises(ValueError) as excinfo:
        da.apply_gufunc(foo, '(i)->()', a, output_dtypes=float, allow_rechunk=False)
    assert 'consists of multiple chunks' in str(excinfo.value)