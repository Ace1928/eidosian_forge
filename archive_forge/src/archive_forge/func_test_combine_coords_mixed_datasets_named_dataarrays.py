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
def test_combine_coords_mixed_datasets_named_dataarrays(self):
    da = DataArray(name='a', data=[4, 5], dims='x', coords={'x': [0, 1]})
    ds = Dataset({'b': ('x', [2, 3])})
    actual = combine_by_coords([da, ds])
    expected = Dataset({'a': ('x', [4, 5]), 'b': ('x', [2, 3])}, coords={'x': ('x', [0, 1])})
    assert_identical(expected, actual)