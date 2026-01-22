from __future__ import annotations
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from xarray import (
from xarray.core import dtypes
from xarray.core.combine import (
from xarray.tests import assert_equal, assert_identical, requires_cftime
from xarray.tests.test_dataset import create_test_data
def test_combine_by_coords_all_unnamed_dataarrays(self):
    unnamed_array = DataArray(data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
    actual = combine_by_coords([unnamed_array])
    expected = unnamed_array
    assert_identical(expected, actual)
    unnamed_array1 = DataArray(data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
    unnamed_array2 = DataArray(data=[3.0, 4.0], coords={'x': [2, 3]}, dims='x')
    actual = combine_by_coords([unnamed_array1, unnamed_array2])
    expected = DataArray(data=[1.0, 2.0, 3.0, 4.0], coords={'x': [0, 1, 2, 3]}, dims='x')
    assert_identical(expected, actual)