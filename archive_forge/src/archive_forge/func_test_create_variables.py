from __future__ import annotations
import copy
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.indexes import (
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_array_equal, assert_identical, requires_cftime
from xarray.tests.test_coding_times import _all_cftime_date_types
def test_create_variables(self) -> None:
    foo_data = np.array([0, 0, 1], dtype='int64')
    bar_data = np.array([1.1, 1.2, 1.3], dtype='float64')
    pd_idx = pd.MultiIndex.from_arrays([foo_data, bar_data], names=('foo', 'bar'))
    index_vars = {'x': IndexVariable('x', pd_idx), 'foo': IndexVariable('x', foo_data, attrs={'unit': 'm'}), 'bar': IndexVariable('x', bar_data, encoding={'fill_value': 0})}
    index = PandasMultiIndex(pd_idx, 'x')
    actual = index.create_variables(index_vars)
    for k, expected in index_vars.items():
        assert_identical(actual[k], expected)
        assert actual[k].dtype == expected.dtype
        if k != 'x':
            assert actual[k].dtype == index.level_coords_dtype[k]