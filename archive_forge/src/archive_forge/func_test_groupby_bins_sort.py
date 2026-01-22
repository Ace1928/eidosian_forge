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
def test_groupby_bins_sort(self) -> None:
    data = xr.DataArray(np.arange(100), dims='x', coords={'x': np.linspace(-100, 100, num=100)})
    binned_mean = data.groupby_bins('x', bins=11).mean()
    assert binned_mean.to_index().is_monotonic_increasing
    with xr.set_options(use_flox=True):
        actual = data.groupby_bins('x', bins=11).count()
    with xr.set_options(use_flox=False):
        expected = data.groupby_bins('x', bins=11).count()
    assert_identical(actual, expected)