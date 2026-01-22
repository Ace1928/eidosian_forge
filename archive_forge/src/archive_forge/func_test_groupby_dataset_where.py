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
def test_groupby_dataset_where() -> None:
    ds = Dataset({'a': ('x', range(5))}, {'c': ('x', [0, 0, 1, 1, 1])})
    cond = Dataset({'a': ('c', [True, False])})
    expected = ds.copy(deep=True)
    expected['a'].values = np.array([0, 1] + [np.nan] * 3)
    actual = ds.groupby('c').where(cond)
    assert_identical(expected, actual)
    ds.attrs['attr'] = 'ds'
    ds.a.attrs['attr'] = 'da'
    actual = ds.groupby('c').where(cond)
    assert actual.attrs == ds.attrs
    assert actual.a.name == 'a'
    assert actual.a.attrs == ds.a.attrs