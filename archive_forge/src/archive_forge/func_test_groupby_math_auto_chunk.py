from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
@requires_dask
def test_groupby_math_auto_chunk() -> None:
    da = xr.DataArray([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dims=('y', 'x'), coords={'label': ('x', [2, 2, 1])})
    sub = xr.DataArray(InaccessibleArray(np.array([1, 2])), dims='label', coords={'label': [1, 2]})
    actual = da.chunk(x=1, y=2).groupby('label') - sub
    assert actual.chunksizes == {'x': (1, 1, 1), 'y': (2, 1)}