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
def test_set_dims(self, dtype):
    array = np.linspace(0, 5, 3 * 10).reshape(3, 10).astype(dtype) * unit_registry.m
    variable = xr.Variable(('x', 'y'), array)
    dims = {'z': 6, 'x': 3, 'a': 1, 'b': 4, 'y': 10}
    expected = attach_units(strip_units(variable).set_dims(dims), extract_units(variable))
    actual = variable.set_dims(dims)
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)