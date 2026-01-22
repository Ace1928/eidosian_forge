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
def test_string_coords(self):
    ds0 = Dataset({'person': ['Alice', 'Bob']})
    ds1 = Dataset({'person': ['Caroline', 'Daniel']})
    expected = {(0,): ds0, (1,): ds1}
    actual, concat_dims = _infer_concat_order_from_coords([ds1, ds0])
    assert_combined_tile_ids_equal(expected, actual)
    assert concat_dims == ['person']