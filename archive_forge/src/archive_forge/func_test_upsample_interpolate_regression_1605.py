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
@requires_scipy
def test_upsample_interpolate_regression_1605(self) -> None:
    dates = pd.date_range('2016-01-01', '2016-03-31', freq='1D')
    expected = xr.DataArray(np.random.random((len(dates), 2, 3)), dims=('time', 'x', 'y'), coords={'time': dates})
    actual = expected.resample(time='1D').interpolate('linear')
    assert_allclose(actual, expected, rtol=1e-16)