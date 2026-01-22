from __future__ import annotations
import functools
import operator
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, duck_array_ops
from xarray.tests import (
from xarray.tests.test_plot import PlotTestCase
from xarray.tests.test_variable import _PAD_XR_NP_ARGS
@pytest.mark.xfail(reason='interpolate_na uses numpy.vectorize')
def test_interpolate_na(self, dtype):
    array1 = np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype) * unit_registry.degK
    array2 = np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype) * unit_registry.Pa
    ds = xr.Dataset({'a': ('x', array1), 'b': ('x', array2)})
    units = extract_units(ds)
    expected = attach_units(strip_units(ds).interpolate_na(dim='x'), units)
    actual = ds.interpolate_na(dim='x')
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)