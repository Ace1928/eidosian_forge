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
def test_combine_concat_over_redundant_nesting(self):
    objs = [[Dataset({'x': [0]}), Dataset({'x': [1]})]]
    actual = combine_nested(objs, concat_dim=[None, 'x'])
    expected = Dataset({'x': [0, 1]})
    assert_identical(expected, actual)
    objs = [[Dataset({'x': [0]})], [Dataset({'x': [1]})]]
    actual = combine_nested(objs, concat_dim=['x', None])
    expected = Dataset({'x': [0, 1]})
    assert_identical(expected, actual)
    objs = [[Dataset({'x': [0]})]]
    actual = combine_nested(objs, concat_dim=[None, None])
    expected = Dataset({'x': [0]})
    assert_identical(expected, actual)