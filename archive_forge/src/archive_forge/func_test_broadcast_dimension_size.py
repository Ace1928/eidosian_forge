from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_broadcast_dimension_size(self):
    actual = merge.broadcast_dimension_size([xr.Variable('x', [1]), xr.Variable('y', [2, 1])])
    assert actual == {'x': 1, 'y': 2}
    actual = merge.broadcast_dimension_size([xr.Variable(('x', 'y'), [[1, 2]]), xr.Variable('y', [2, 1])])
    assert actual == {'x': 1, 'y': 2}
    with pytest.raises(ValueError):
        merge.broadcast_dimension_size([xr.Variable(('x', 'y'), [[1, 2]]), xr.Variable('y', [2])])