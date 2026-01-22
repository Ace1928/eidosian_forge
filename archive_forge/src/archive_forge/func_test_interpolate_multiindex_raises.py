from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
def test_interpolate_multiindex_raises():
    data = np.random.randn(2, 3)
    data[1, 1] = np.nan
    da = xr.DataArray(data, coords=[('x', ['a', 'b']), ('y', [0, 1, 2])])
    das = da.stack(z=('x', 'y'))
    with pytest.raises(TypeError, match="Index 'z' must be castable to float64"):
        das.interpolate_na(dim='z')