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
def test_upsample(self) -> None:
    times = pd.date_range('2000-01-01', freq='6h', periods=5)
    array = DataArray(np.arange(5), [('time', times)])
    actual = array.resample(time='3h').ffill()
    expected = DataArray(array.to_series().resample('3h').ffill())
    assert_identical(expected, actual)
    actual = array.resample(time='3h').bfill()
    expected = DataArray(array.to_series().resample('3h').bfill())
    assert_identical(expected, actual)
    actual = array.resample(time='3h').asfreq()
    expected = DataArray(array.to_series().resample('3h').asfreq())
    assert_identical(expected, actual)
    actual = array.resample(time='3h').pad()
    expected = DataArray(array.to_series().resample('3h').ffill())
    assert_identical(expected, actual)
    rs = array.resample(time='3h')
    actual = rs.nearest()
    new_times = rs.groupers[0].full_index
    expected = DataArray(array.reindex(time=new_times, method='nearest'))
    assert_identical(expected, actual)