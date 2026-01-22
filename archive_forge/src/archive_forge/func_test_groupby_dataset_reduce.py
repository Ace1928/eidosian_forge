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
def test_groupby_dataset_reduce() -> None:
    data = Dataset({'xy': (['x', 'y'], np.random.randn(3, 4)), 'xonly': ('x', np.random.randn(3)), 'yonly': ('y', np.random.randn(4)), 'letters': ('y', ['a', 'a', 'b', 'b'])})
    expected = data.mean('y')
    expected['yonly'] = expected['yonly'].variable.set_dims({'x': 3})
    actual = data.groupby('x').mean(...)
    assert_allclose(expected, actual)
    actual = data.groupby('x').mean('y')
    assert_allclose(expected, actual)
    letters = data['letters']
    expected = Dataset({'xy': data['xy'].groupby(letters).mean(...), 'xonly': data['xonly'].mean().variable.set_dims({'letters': 2}), 'yonly': data['yonly'].groupby(letters).mean()})
    actual = data.groupby('letters').mean(...)
    assert_allclose(expected, actual)