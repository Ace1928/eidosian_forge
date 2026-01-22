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
def test_concat_one_dim_merge_another(self):
    data = create_test_data(add_attrs=False)
    data1 = data.copy(deep=True)
    data2 = data.copy(deep=True)
    objs = [[data1.var1.isel(dim2=slice(4)), data2.var1.isel(dim2=slice(4, 9))], [data1.var2.isel(dim2=slice(4)), data2.var2.isel(dim2=slice(4, 9))]]
    expected = data[['var1', 'var2']]
    actual = combine_nested(objs, concat_dim=[None, 'dim2'])
    assert_identical(expected, actual)