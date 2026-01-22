from __future__ import annotations
import pytest
import dask.array as da
from dask.array.utils import assert_eq
def test_asarray_xarray_intersphinx_workaround():
    module = xr.DataArray.__module__
    try:
        xr.DataArray.__module__ = 'xarray'
        y = da.asarray(xr.DataArray([1, 2, 3.0]))
        assert isinstance(y, da.Array)
        assert type(y._meta).__name__ == 'ndarray'
        assert_eq(y, y)
    finally:
        xr.DataArray.__module__ = module