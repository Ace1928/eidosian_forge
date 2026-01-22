from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_allclose, assert_array_equal, mock
from xarray.tests import assert_identical as assert_identical_
def test_xarray_defers_to_unrecognized_type():

    class Other:

        def __array_ufunc__(self, *args, **kwargs):
            return 'other'
    xarray_obj = xr.DataArray([1, 2, 3])
    other = Other()
    assert np.maximum(xarray_obj, other) == 'other'
    assert np.sin(xarray_obj, out=other) == 'other'