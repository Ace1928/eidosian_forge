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
@pytest.mark.parametrize('keep_attrs', [None, True, False])
def test_groupby_keep_attrs(self, keep_attrs: bool | None) -> None:
    array = self.da
    array.attrs['foo'] = 'bar'
    with xr.set_options(use_flox=False):
        expected = array.groupby('abc').mean(keep_attrs=keep_attrs)
    with xr.set_options(use_flox=True):
        actual = array.groupby('abc').mean(keep_attrs=keep_attrs)
    actual.data = expected.data
    assert_identical(expected, actual)