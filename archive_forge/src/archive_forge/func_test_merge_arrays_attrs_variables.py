from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
@pytest.mark.parametrize('combine_attrs, attrs1, attrs2, expected_attrs, expect_exception', [('no_conflicts', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {}, {'a': 1, 'b': 2}, False), ('no_conflicts', {}, {'a': 1, 'c': 3}, {'a': 1, 'c': 3}, False), ('no_conflicts', {'a': 1, 'b': 2}, {'a': 4, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}, True), ('drop', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}, False), ('identical', {'a': 1, 'b': 2}, {'a': 1, 'c': 3}, {'a': 1, 'b': 2}, True), ('override', {'a': 1, 'b': 2}, {'a': 4, 'b': 5, 'c': 3}, {'a': 1, 'b': 2}, False), ('drop_conflicts', {'a': 1, 'b': 2, 'c': 3}, {'b': 1, 'c': 3, 'd': 4}, {'a': 1, 'c': 3, 'd': 4}, False), (lambda attrs, context: attrs[1], {'a': 1, 'b': 2, 'c': 3}, {'a': 4, 'b': 3, 'c': 1}, {'a': 4, 'b': 3, 'c': 1}, False)])
def test_merge_arrays_attrs_variables(self, combine_attrs, attrs1, attrs2, expected_attrs, expect_exception):
    """check that combine_attrs is used on data variables and coords"""
    data1 = xr.Dataset({'var1': ('dim1', [], attrs1)}, coords={'dim1': ('dim1', [], attrs1)})
    data2 = xr.Dataset({'var1': ('dim1', [], attrs2)}, coords={'dim1': ('dim1', [], attrs2)})
    if expect_exception:
        with pytest.raises(MergeError, match='combine_attrs'):
            actual = xr.merge([data1, data2], combine_attrs=combine_attrs)
    else:
        actual = xr.merge([data1, data2], combine_attrs=combine_attrs)
        expected = xr.Dataset({'var1': ('dim1', [], expected_attrs)}, coords={'dim1': ('dim1', [], expected_attrs)})
        assert_identical(actual, expected)