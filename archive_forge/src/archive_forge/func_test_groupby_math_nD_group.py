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
def test_groupby_math_nD_group() -> None:
    N = 40
    da = DataArray(np.random.random((N, N)), dims=('x', 'y'), coords={'labels': ('x', np.repeat(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], repeats=N // 8))})
    da['labels2d'] = xr.broadcast(da.labels, da)[0]
    g = da.groupby('labels2d')
    mean = g.mean()
    expected = da - mean.sel(labels2d=da.labels2d)
    expected['labels'] = expected.labels.broadcast_like(expected.labels2d)
    actual = g - mean
    assert_identical(expected, actual)
    da['num'] = ('x', np.repeat([1, 2, 3, 4, 5, 6, 7, 8], repeats=N // 8))
    da['num2d'] = xr.broadcast(da.num, da)[0]
    g = da.groupby_bins('num2d', bins=[0, 4, 6])
    mean = g.mean()
    idxr = np.digitize(da.num2d, bins=(0, 4, 6), right=True)[:30, :] - 1
    expanded_mean = mean.drop_vars('num2d_bins').isel(num2d_bins=(('x', 'y'), idxr))
    expected = da.isel(x=slice(30)) - expanded_mean
    expected['labels'] = expected.labels.broadcast_like(expected.labels2d)
    expected['num'] = expected.num.broadcast_like(expected.num2d)
    expected['num2d_bins'] = (('x', 'y'), mean.num2d_bins.data[idxr])
    actual = g - mean
    assert_identical(expected, actual)