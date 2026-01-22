from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
@pytest.mark.parametrize('func,sparse_output', [(do('squeeze'), True), param(do('searchsorted', [1, 2, 3]), False, marks=xfail(reason="'COO' object has no attribute 'searchsorted'"))])
def test_datarray_1d_method(func, sparse_output):
    arr_s = make_xrarray({'x': 10}, coords={'x': np.arange(10)})
    arr_d = xr.DataArray(arr_s.data.todense(), coords=arr_s.coords, dims=arr_s.dims)
    ret_s = func(arr_s)
    ret_d = func(arr_d)
    if sparse_output:
        assert isinstance(ret_s.data, sparse.SparseArray)
        assert np.allclose(ret_s.data.todense(), ret_d.data, equal_nan=True)
    else:
        assert np.allclose(ret_s, ret_d, equal_nan=True)