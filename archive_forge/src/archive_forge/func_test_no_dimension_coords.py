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
def test_no_dimension_coords(self):
    ds0 = Dataset({'foo': ('x', [0, 1])})
    ds1 = Dataset({'foo': ('x', [2, 3])})
    with pytest.raises(ValueError, match='Could not find any dimension'):
        _infer_concat_order_from_coords([ds1, ds0])