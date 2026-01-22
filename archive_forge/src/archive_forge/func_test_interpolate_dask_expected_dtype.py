from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_dask
@requires_scipy
@pytest.mark.parametrize('dtype, method', [(int, 'linear'), (int, 'nearest')])
def test_interpolate_dask_expected_dtype(dtype, method):
    da = xr.DataArray(data=np.array([0, 1], dtype=dtype), dims=['time'], coords=dict(time=np.array([0, 1]))).chunk(dict(time=2))
    da = da.interp(time=np.array([0, 0.5, 1, 2]), method=method)
    assert da.dtype == da.compute().dtype