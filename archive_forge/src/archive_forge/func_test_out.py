from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_allclose, assert_array_equal, mock
from xarray.tests import assert_identical as assert_identical_
def test_out():
    xarray_obj = xr.DataArray([1, 2, 3])
    with pytest.raises(NotImplementedError, match='`out` argument'):
        np.add(xarray_obj, 1, out=xarray_obj)
    other = np.zeros((3,))
    np.add(other, xarray_obj, out=other)
    assert_identical(other, np.array([1, 2, 3]))