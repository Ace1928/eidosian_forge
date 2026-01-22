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
@requires_dask
def test_array_scalar_format(self) -> None:
    var = xr.DataArray(np.array(0))
    assert format(var, '') == repr(var)
    assert format(var, 'd') == '0'
    assert format(var, '.2f') == '0.00'
    import dask.array as da
    var = xr.DataArray(da.array(0))
    assert format(var, '') == repr(var)
    with pytest.raises(TypeError) as excinfo:
        format(var, '.2f')
    assert 'unsupported format string passed to' in str(excinfo.value)
    var = xr.DataArray([0.1, 0.2])
    with pytest.raises(NotImplementedError) as excinfo:
        format(var, '.2f')
    assert 'Using format_spec is only supported' in str(excinfo.value)