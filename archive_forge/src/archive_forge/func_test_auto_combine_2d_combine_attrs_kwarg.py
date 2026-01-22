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
def test_auto_combine_2d_combine_attrs_kwarg(self):
    ds = lambda x: create_test_data(x, add_attrs=False)
    partway1 = concat([ds(0), ds(3)], dim='dim1')
    partway2 = concat([ds(1), ds(4)], dim='dim1')
    partway3 = concat([ds(2), ds(5)], dim='dim1')
    expected = concat([partway1, partway2, partway3], dim='dim2')
    expected_dict = {}
    expected_dict['drop'] = expected.copy(deep=True)
    expected_dict['drop'].attrs = {}
    expected_dict['no_conflicts'] = expected.copy(deep=True)
    expected_dict['no_conflicts'].attrs = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
    expected_dict['override'] = expected.copy(deep=True)
    expected_dict['override'].attrs = {'a': 1}
    f = lambda attrs, context: attrs[0]
    expected_dict[f] = expected.copy(deep=True)
    expected_dict[f].attrs = f([{'a': 1}], None)
    datasets = [[ds(0), ds(1), ds(2)], [ds(3), ds(4), ds(5)]]
    datasets[0][0].attrs = {'a': 1}
    datasets[0][1].attrs = {'a': 1, 'b': 2}
    datasets[0][2].attrs = {'a': 1, 'c': 3}
    datasets[1][0].attrs = {'a': 1, 'd': 4}
    datasets[1][1].attrs = {'a': 1, 'e': 5}
    datasets[1][2].attrs = {'a': 1, 'f': 6}
    with pytest.raises(ValueError, match="combine_attrs='identical'"):
        result = combine_nested(datasets, concat_dim=['dim1', 'dim2'], combine_attrs='identical')
    for combine_attrs in expected_dict:
        result = combine_nested(datasets, concat_dim=['dim1', 'dim2'], combine_attrs=combine_attrs)
        assert_identical(result, expected_dict[combine_attrs])