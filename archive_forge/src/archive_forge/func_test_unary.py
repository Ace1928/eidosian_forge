from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_allclose, assert_array_equal, mock
from xarray.tests import assert_identical as assert_identical_
@pytest.mark.parametrize('a', [xr.Variable(['x'], [0, 0]), xr.DataArray([0, 0], dims='x'), xr.Dataset({'y': ('x', [0, 0])})])
def test_unary(a):
    assert_allclose(a + 1, np.cos(a))