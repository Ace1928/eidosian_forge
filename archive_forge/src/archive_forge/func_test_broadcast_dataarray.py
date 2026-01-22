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
def test_broadcast_dataarray(dtype):
    array1 = np.linspace(0, 10, 2) * unit_registry.Pa
    array2 = np.linspace(0, 10, 3) * unit_registry.Pa
    a = xr.DataArray(data=array1, dims='x')
    b = xr.DataArray(data=array2, dims='y')
    units_a = extract_units(a)
    units_b = extract_units(b)
    expected_a, expected_b = xr.broadcast(strip_units(a), strip_units(b))
    expected_a = attach_units(expected_a, units_a)
    expected_b = convert_units(attach_units(expected_b, units_a), units_b)
    actual_a, actual_b = xr.broadcast(a, b)
    assert_units_equal(expected_a, actual_a)
    assert_identical(expected_a, actual_a)
    assert_units_equal(expected_b, actual_b)
    assert_identical(expected_b, actual_b)