from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
def test_array_repr_recursive(self) -> None:
    var = xr.Variable('x', [0, 1])
    var.attrs['x'] = var
    formatting.array_repr(var)
    da = xr.DataArray([0, 1], dims=['x'])
    da.attrs['x'] = da
    formatting.array_repr(da)
    var.attrs['x'] = da
    da.attrs['x'] = var
    formatting.array_repr(var)
    formatting.array_repr(da)