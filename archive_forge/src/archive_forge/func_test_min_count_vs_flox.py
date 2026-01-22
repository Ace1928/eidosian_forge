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
@pytest.mark.parametrize('func', ['sum', 'prod'])
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.parametrize('min_count', [None, 1])
def test_min_count_vs_flox(func: str, min_count: int | None, skipna: bool) -> None:
    da = DataArray(data=np.array([np.nan, 1, 1, np.nan, 1, 1]), dims='x', coords={'labels': ('x', np.array([1, 2, 3, 1, 2, 3]))})
    gb = da.groupby('labels')
    method = operator.methodcaller(func, min_count=min_count, skipna=skipna)
    with xr.set_options(use_flox=True):
        actual = method(gb)
    with xr.set_options(use_flox=False):
        expected = method(gb)
    assert_identical(actual, expected)