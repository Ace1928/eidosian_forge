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
def test_groupby_dataset_math(squeeze: bool) -> None:

    def reorder_dims(x):
        return x.transpose('dim1', 'dim2', 'dim3', 'time')
    ds = create_test_data()
    ds['dim1'] = ds['dim1']
    grouped = ds.groupby('dim1', squeeze=squeeze)
    expected = reorder_dims(ds + ds.coords['dim1'])
    actual = grouped + ds.coords['dim1']
    assert_identical(expected, reorder_dims(actual))
    actual = ds.coords['dim1'] + grouped
    assert_identical(expected, reorder_dims(actual))
    ds2 = 2 * ds
    expected = reorder_dims(ds + ds2)
    actual = grouped + ds2
    assert_identical(expected, reorder_dims(actual))
    actual = ds2 + grouped
    assert_identical(expected, reorder_dims(actual))