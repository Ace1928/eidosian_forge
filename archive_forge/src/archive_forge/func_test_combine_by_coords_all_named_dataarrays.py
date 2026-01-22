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
def test_combine_by_coords_all_named_dataarrays(self):
    named_da = DataArray(name='a', data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
    actual = combine_by_coords([named_da])
    expected = named_da.to_dataset()
    assert_identical(expected, actual)
    named_da1 = DataArray(name='a', data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x')
    named_da2 = DataArray(name='b', data=[3.0, 4.0], coords={'x': [2, 3]}, dims='x')
    actual = combine_by_coords([named_da1, named_da2])
    expected = Dataset({'a': DataArray(data=[1.0, 2.0], coords={'x': [0, 1]}, dims='x'), 'b': DataArray(data=[3.0, 4.0], coords={'x': [2, 3]}, dims='x')})
    assert_identical(expected, actual)