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
def test_groupby_indexvariable(use_flox: bool) -> None:
    array = xr.DataArray([1, 2, 3], [('x', [2, 2, 1])])
    iv = xr.IndexVariable(dims='x', data=pd.Index(array.x.values))
    with xr.set_options(use_flox=use_flox):
        actual = array.groupby(iv).sum()
    actual = array.groupby(iv).sum()
    expected = xr.DataArray([3, 3], [('x', [1, 2])])
    assert_identical(expected, actual)