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
def test_ds_resample_apply_func_args(self) -> None:

    def func(arg1, arg2, arg3=0.0):
        return arg1.mean('time') + arg2 + arg3
    times = pd.date_range('2000', freq='D', periods=3)
    ds = xr.Dataset({'foo': ('time', [1.0, 1.0, 1.0]), 'time': times})
    expected = xr.Dataset({'foo': ('time', [3.0, 3.0, 3.0]), 'time': times})
    actual = ds.resample(time='D').map(func, args=(1.0,), arg3=1.0)
    assert_identical(expected, actual)