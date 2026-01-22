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
def test_concat_multiple_dims(self):
    objs = [[Dataset({'a': (('x', 'y'), [[0]])}), Dataset({'a': (('x', 'y'), [[1]])})], [Dataset({'a': (('x', 'y'), [[2]])}), Dataset({'a': (('x', 'y'), [[3]])})]]
    actual = combine_nested(objs, concat_dim=['x', 'y'])
    expected = Dataset({'a': (('x', 'y'), [[0, 1], [2, 3]])})
    assert_identical(expected, actual)