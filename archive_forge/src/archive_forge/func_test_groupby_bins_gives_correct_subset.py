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
@pytest.mark.parametrize('use_flox', [True, False])
def test_groupby_bins_gives_correct_subset(self, use_flox: bool) -> None:
    rng = np.random.default_rng(42)
    coords = rng.normal(5, 5, 1000)
    bins = np.logspace(-4, 1, 10)
    labels = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    darr = xr.DataArray(coords, coords=[coords], dims=['coords'])
    expected = xr.DataArray([np.nan, np.nan, 1, 1, 1, 8, 31, 104, 542], dims='coords_bins', coords={'coords_bins': labels})
    gb = darr.groupby_bins('coords', bins, labels=labels)
    with xr.set_options(use_flox=use_flox):
        actual = gb.count()
    assert_identical(actual, expected)