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
@pytest.mark.parametrize('squeeze', [True, False])
def test_groupby_math_squeeze(self, squeeze: bool) -> None:
    array = self.da
    grouped = array.groupby('x', squeeze=squeeze)
    expected = array + array.coords['x']
    actual = grouped + array.coords['x']
    assert_identical(expected, actual)
    actual = array.coords['x'] + grouped
    assert_identical(expected, actual)
    ds = array.coords['x'].to_dataset(name='X')
    expected = array + ds
    actual = grouped + ds
    assert_identical(expected, actual)
    actual = ds + grouped
    assert_identical(expected, actual)