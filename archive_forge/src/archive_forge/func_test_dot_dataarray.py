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
def test_dot_dataarray(dtype):
    array1 = np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * unit_registry.m / unit_registry.s
    array2 = np.linspace(10, 20, 10 * 20).reshape(10, 20).astype(dtype) * unit_registry.s
    data_array = xr.DataArray(data=array1, dims=('x', 'y'))
    other = xr.DataArray(data=array2, dims=('y', 'z'))
    with xr.set_options(use_opt_einsum=False):
        expected = attach_units(xr.dot(strip_units(data_array), strip_units(other)), {None: unit_registry.m})
        actual = xr.dot(data_array, other)
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)