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
@pytest.mark.parametrize('combine_attrs, attrs1, attrs2, expected_attrs, expect_exception', [('no_conflicts', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {}, {'a': 1, 'b': 2}, False), ('no_conflicts', {}, {'a': 1, 'c': 3}, {'a': 1, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {'a': 4, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, True), ('drop', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2}, True), ('override', {'a': 1, 'b': 2}, {'a': 4, 'b': 5, 'c': 3}, {'a': 1, 'b': 2}, False), ('drop_conflicts', {'a': 1, 'b': 2, 'c': 3}, {'b': 1, 'c': 3, 'd': 4}, {'a': 1, 'c': 3, 'd': 4}, False)])
def test_combine_nested_combine_attrs_variables(self, combine_attrs, attrs1, attrs2, expected_attrs, expect_exception):
    """check that combine_attrs is used on data variables and coords"""
    data1 = Dataset({'a': ('x', [1, 2], attrs1), 'b': ('x', [3, -1], attrs1), 'x': ('x', [0, 1], attrs1)})
    data2 = Dataset({'a': ('x', [2, 3], attrs2), 'b': ('x', [-2, 1], attrs2), 'x': ('x', [2, 3], attrs2)})
    if expect_exception:
        with pytest.raises(MergeError, match='combine_attrs'):
            combine_by_coords([data1, data2], combine_attrs=combine_attrs)
    else:
        actual = combine_by_coords([data1, data2], combine_attrs=combine_attrs)
        expected = Dataset({'a': ('x', [1, 2, 2, 3], expected_attrs), 'b': ('x', [3, -1, -2, 1], expected_attrs)}, {'x': ('x', [0, 1, 2, 3], expected_attrs)})
        assert_identical(actual, expected)