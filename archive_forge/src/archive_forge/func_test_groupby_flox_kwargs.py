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
@requires_flox
@pytest.mark.parametrize('kwargs', [{'method': 'map-reduce'}, {'engine': 'numpy'}])
def test_groupby_flox_kwargs(kwargs) -> None:
    ds = Dataset({'a': ('x', range(5))}, {'c': ('x', [0, 0, 1, 1, 1])})
    with xr.set_options(use_flox=False):
        expected = ds.groupby('c').mean()
    with xr.set_options(use_flox=True):
        actual = ds.groupby('c').mean(**kwargs)
    assert_identical(expected, actual)