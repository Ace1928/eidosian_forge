from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, merge
from xarray.core.merge import MergeError
from xarray.testing import assert_equal, assert_identical
from xarray.tests.test_dataset import create_test_data
def test_merge_no_conflicts_multi_var(self):
    data = create_test_data(add_attrs=False)
    data1 = data.copy(deep=True)
    data2 = data.copy(deep=True)
    expected = data[['var1', 'var2']]
    actual = xr.merge([data1.var1, data2.var2], compat='no_conflicts')
    assert_identical(expected, actual)
    data1['var1'][:, :5] = np.nan
    data2['var1'][:, 5:] = np.nan
    data1['var2'][:4, :] = np.nan
    data2['var2'][4:, :] = np.nan
    del data2['var3']
    actual = xr.merge([data1, data2], compat='no_conflicts')
    assert_equal(data, actual)