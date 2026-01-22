from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_scipy
def test_interpolate_methods():
    for method in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']:
        kwargs = {}
        da = xr.DataArray(np.array([0, 1, 2, np.nan, np.nan, np.nan, 6, 7, 8], dtype=np.float64), dims='x')
        actual = da.interpolate_na('x', method=method, **kwargs)
        assert actual.isnull().sum() == 0
        actual = da.interpolate_na('x', method=method, limit=2, **kwargs)
        assert actual.isnull().sum() == 1