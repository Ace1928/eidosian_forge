from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
@pytest.mark.parametrize('combine_attrs, attrs1, attrs2, expected_attrs, expect_exception', [('no_conflicts', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {}, {'a': 1, 'b': 2}, False), ('no_conflicts', {}, {'a': 1, 'c': 3}, {'a': 1, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {'a': 4, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, True), ('drop', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2}, True), ('override', {'a': 1, 'b': 2}, {'a': 4, 'b': 5, 'c': 3}, {'a': 1, 'b': 2}, False), ('drop_conflicts', {'a': 41, 'b': 42, 'c': 43}, {'b': 2, 'c': 43, 'd': 44}, {'a': 41, 'c': 43, 'd': 44}, False), (lambda attrs, context: {'a': -1, 'b': 0, 'c': 1} if any(attrs) else {}, {'a': 41, 'b': 42, 'c': 43}, {'b': 2, 'c': 43, 'd': 44}, {'a': -1, 'b': 0, 'c': 1}, False)])
def test_concat_combine_attrs_kwarg_variables(self, combine_attrs, attrs1, attrs2, expected_attrs, expect_exception):
    """check that combine_attrs is used on data variables and coords"""
    ds1 = Dataset({'a': ('x', [0], attrs1)}, coords={'x': ('x', [0], attrs1)})
    ds2 = Dataset({'a': ('x', [0], attrs2)}, coords={'x': ('x', [1], attrs2)})
    if expect_exception:
        with pytest.raises(ValueError, match=f"combine_attrs='{combine_attrs}'"):
            concat([ds1, ds2], dim='x', combine_attrs=combine_attrs)
    else:
        actual = concat([ds1, ds2], dim='x', combine_attrs=combine_attrs)
        expected = Dataset({'a': ('x', [0, 0], expected_attrs)}, {'x': ('x', [0, 1], expected_attrs)})
        assert_identical(actual, expected)