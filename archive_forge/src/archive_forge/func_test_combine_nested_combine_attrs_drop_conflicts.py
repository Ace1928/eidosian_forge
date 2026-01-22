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
def test_combine_nested_combine_attrs_drop_conflicts(self):
    objs = [Dataset({'x': [0], 'y': [0]}, attrs={'a': 1, 'b': 2, 'c': 3}), Dataset({'x': [1], 'y': [1]}, attrs={'a': 1, 'b': 0, 'd': 3})]
    expected = Dataset({'x': [0, 1], 'y': [0, 1]}, attrs={'a': 1, 'c': 3, 'd': 3})
    actual = combine_nested(objs, concat_dim='x', join='outer', combine_attrs='drop_conflicts')
    assert_identical(expected, actual)