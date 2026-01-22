from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
def test_interpolate_duplicate_values_raises():
    data = np.random.randn(2, 3)
    da = xr.DataArray(data, coords=[('x', ['a', 'a']), ('y', [0, 1, 2])])
    with pytest.raises(ValueError, match="Index 'x' has duplicate values"):
        da.interpolate_na(dim='x', method='foo')