from __future__ import annotations
import warnings
import numpy as np
import pytest
import xarray as xr
from xarray.tests import has_dask
def test_allclose_regression() -> None:
    x = xr.DataArray(1.01)
    y = xr.DataArray(1.02)
    xr.testing.assert_allclose(x, y, atol=0.01)