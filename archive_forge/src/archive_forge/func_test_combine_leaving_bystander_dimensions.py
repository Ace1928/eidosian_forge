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
def test_combine_leaving_bystander_dimensions(self):
    ycoord = ['a', 'c', 'b']
    data = np.random.rand(7, 3)
    ds1 = Dataset(data_vars=dict(data=(['x', 'y'], data[:3, :])), coords=dict(x=[1, 2, 3], y=ycoord))
    ds2 = Dataset(data_vars=dict(data=(['x', 'y'], data[3:, :])), coords=dict(x=[4, 5, 6, 7], y=ycoord))
    expected = Dataset(data_vars=dict(data=(['x', 'y'], data)), coords=dict(x=[1, 2, 3, 4, 5, 6, 7], y=ycoord))
    actual = combine_by_coords((ds1, ds2))
    assert_identical(expected, actual)