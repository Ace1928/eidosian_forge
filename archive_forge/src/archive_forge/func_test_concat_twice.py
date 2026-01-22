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
@pytest.mark.parametrize('concat_dim', ['dim1', 'new_dim'])
def test_concat_twice(self, create_combined_ids, concat_dim):
    shape = (2, 3)
    combined_ids = create_combined_ids(shape)
    result = _combine_nd(combined_ids, concat_dims=['dim1', concat_dim])
    ds = create_test_data
    partway1 = concat([ds(0), ds(3)], dim='dim1')
    partway2 = concat([ds(1), ds(4)], dim='dim1')
    partway3 = concat([ds(2), ds(5)], dim='dim1')
    expected = concat([partway1, partway2, partway3], dim=concat_dim)
    assert_equal(result, expected)