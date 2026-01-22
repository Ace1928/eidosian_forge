from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
@requires_numbagg
@requires_dask
def test_ffill_use_numbagg_dask():
    with xr.set_options(use_bottleneck=False):
        da = xr.DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims='x')
        da = da.chunk(x=-1)
        _ = da.ffill('x').compute()