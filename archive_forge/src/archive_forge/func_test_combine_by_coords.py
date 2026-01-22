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
def test_combine_by_coords(self):
    objs = [Dataset({'x': [0]}), Dataset({'x': [1]})]
    actual = combine_by_coords(objs)
    expected = Dataset({'x': [0, 1]})
    assert_identical(expected, actual)
    actual = combine_by_coords([actual])
    assert_identical(expected, actual)
    objs = [Dataset({'x': [0, 1]}), Dataset({'x': [2]})]
    actual = combine_by_coords(objs)
    expected = Dataset({'x': [0, 1, 2]})
    assert_identical(expected, actual)
    objs = [Dataset({'x': ('a', [0]), 'y': ('a', [0]), 'a': [0]}), Dataset({'x': ('a', [1]), 'y': ('a', [1]), 'a': [1]})]
    actual = combine_by_coords(objs)
    expected = Dataset({'x': ('a', [0, 1]), 'y': ('a', [0, 1]), 'a': [0, 1]})
    assert_identical(expected, actual)
    objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'y': [1], 'x': [1]})]
    actual = combine_by_coords(objs)
    expected = Dataset({'x': [0, 1], 'y': [0, 1]})
    assert_equal(actual, expected)
    objs = [Dataset({'x': 0}), Dataset({'x': 1})]
    with pytest.raises(ValueError, match='Could not find any dimension coordinates'):
        combine_by_coords(objs)
    objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'x': [0]})]
    with pytest.raises(ValueError, match='Every dimension needs a coordinate'):
        combine_by_coords(objs)