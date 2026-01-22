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
@pytest.mark.parametrize('compute_backend', ['numbagg', None], indirect=True)
@pytest.mark.parametrize('func', (method('diff', dim='x'), method('differentiate', coord='x'), method('integrate', coord='x'), method('quantile', q=[0.25, 0.75]), method('reduce', func=np.sum, dim='x'), method('map', np.fabs)), ids=repr)
@pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
def test_computation(self, func, variant, dtype, compute_backend):
    variants = {'data': ((unit_registry.degK, unit_registry.Pa), 1, 1), 'dims': ((1, 1), unit_registry.m, 1), 'coords': ((1, 1), 1, unit_registry.m)}
    (unit1, unit2), dim_unit, coord_unit = variants.get(variant)
    array1 = np.linspace(-5, 5, 4 * 5).reshape(4, 5).astype(dtype) * unit1
    array2 = np.linspace(10, 20, 4 * 3).reshape(4, 3).astype(dtype) * unit2
    x = np.arange(4) * dim_unit
    y = np.arange(5) * dim_unit
    z = np.arange(3) * dim_unit
    ds = xr.Dataset(data_vars={'a': xr.DataArray(data=array1, dims=('x', 'y')), 'b': xr.DataArray(data=array2, dims=('x', 'z'))}, coords={'x': x, 'y': y, 'z': z, 'y2': ('y', np.arange(5) * coord_unit)})
    units = extract_units(ds)
    expected = attach_units(func(strip_units(ds)), units)
    actual = func(ds)
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)