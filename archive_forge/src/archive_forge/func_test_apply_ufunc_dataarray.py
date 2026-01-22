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
@pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
def test_apply_ufunc_dataarray(variant, dtype):
    variants = {'data': (unit_registry.m, 1, 1), 'dims': (1, unit_registry.m, 1), 'coords': (1, 1, unit_registry.m)}
    data_unit, dim_unit, coord_unit = variants.get(variant)
    func = functools.partial(xr.apply_ufunc, np.mean, input_core_dims=[['x']], kwargs={'axis': -1})
    array = np.linspace(0, 10, 20).astype(dtype) * data_unit
    x = np.arange(20) * dim_unit
    u = np.linspace(-1, 1, 20) * coord_unit
    data_array = xr.DataArray(data=array, dims='x', coords={'x': x, 'u': ('x', u)})
    expected = attach_units(func(strip_units(data_array)), extract_units(data_array))
    actual = func(data_array)
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)