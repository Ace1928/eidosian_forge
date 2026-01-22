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
def test_concat_errors(self):
    data = create_test_data()
    split_data = [data.isel(dim1=slice(3)), data.isel(dim1=slice(3, None))]
    with pytest.raises(ValueError, match='must supply at least one'):
        concat([], 'dim1')
    with pytest.raises(ValueError, match="Cannot specify both .*='different'"):
        concat([data, data], dim='concat_dim', data_vars='different', compat='override')
    with pytest.raises(ValueError, match='must supply at least one'):
        concat([], 'dim1')
    with pytest.raises(ValueError, match='are not found in the coordinates'):
        concat([data, data], 'new_dim', coords=['not_found'])
    with pytest.raises(ValueError, match='are not found in the data variables'):
        concat([data, data], 'new_dim', data_vars=['not_found'])
    with pytest.raises(ValueError, match='global attributes not'):
        data0 = deepcopy(split_data[0])
        data1 = deepcopy(split_data[1])
        data1.attrs['foo'] = 'bar'
        concat([data0, data1], 'dim1', compat='identical')
    assert_identical(data, concat([data0, data1], 'dim1', compat='equals'))
    with pytest.raises(ValueError, match='compat.* invalid'):
        concat(split_data, 'dim1', compat='foobar')
    with pytest.raises(ValueError, match='unexpected value for'):
        concat([data, data], 'new_dim', coords='foobar')
    with pytest.raises(ValueError, match='coordinate in some datasets but not others'):
        concat([Dataset({'x': 0}), Dataset({'x': [1]})], dim='z')
    with pytest.raises(ValueError, match='coordinate in some datasets but not others'):
        concat([Dataset({'x': 0}), Dataset({}, {'x': 1})], dim='z')