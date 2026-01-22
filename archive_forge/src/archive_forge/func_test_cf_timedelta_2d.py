from __future__ import annotations
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime
from xarray import (
from xarray.coding.times import (
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.core.utils import is_duck_dask_array
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
def test_cf_timedelta_2d() -> None:
    units = 'days'
    numbers = np.atleast_2d([1, 2, 3])
    timedeltas = np.atleast_2d(to_timedelta_unboxed(['1D', '2D', '3D']))
    expected = timedeltas
    actual = coding.times.decode_cf_timedelta(numbers, units)
    assert_array_equal(expected, actual)
    assert expected.dtype == actual.dtype