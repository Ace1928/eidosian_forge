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
def test_multi_index_groupby_sum() -> None:
    ds = xr.Dataset({'foo': (('x', 'y', 'z'), np.ones((3, 4, 2)))}, {'x': ['a', 'b', 'c'], 'y': [1, 2, 3, 4]})
    expected = ds.sum('z')
    actual = ds.stack(space=['x', 'y']).groupby('space').sum('z').unstack('space')
    assert_equal(expected, actual)