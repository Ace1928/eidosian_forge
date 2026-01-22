from __future__ import annotations
import itertools
from typing import Any
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable
from xarray.core import indexing, nputils
from xarray.core.indexes import PandasIndex, PandasMultiIndex
from xarray.core.types import T_Xarray
from xarray.tests import (
@requires_dask
def test_indexing_dask_array():
    import dask.array
    da = DataArray(np.ones(10 * 3 * 3).reshape((10, 3, 3)), dims=('time', 'x', 'y')).chunk(dict(time=-1, x=1, y=1))
    with raise_if_dask_computes():
        actual = da.isel(time=dask.array.from_array([9], chunks=(1,)))
    expected = da.isel(time=[9])
    assert_identical(actual, expected)