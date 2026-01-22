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
def test_groupby_dataset_assign() -> None:
    ds = Dataset({'a': ('x', range(3))}, {'b': ('x', ['A'] * 2 + ['B'])})
    actual = ds.groupby('b').assign(c=lambda ds: 2 * ds.a)
    expected = ds.merge({'c': ('x', [0, 2, 4])})
    assert_identical(actual, expected)
    actual = ds.groupby('b').assign(c=lambda ds: ds.a.sum())
    expected = ds.merge({'c': ('x', [1, 1, 2])})
    assert_identical(actual, expected)
    actual = ds.groupby('b').assign_coords(c=lambda ds: ds.a.sum())
    expected = expected.set_coords('c')
    assert_identical(actual, expected)