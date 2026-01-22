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
@pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.mm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')), ids=repr)
@pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
def test_merge_dataarray(variant, unit, error, dtype):
    original_unit = unit_registry.m
    variants = {'data': ((original_unit, unit), (1, 1), (1, 1)), 'dims': ((1, 1), (original_unit, unit), (1, 1)), 'coords': ((1, 1), (1, 1), (original_unit, unit))}
    (data_unit1, data_unit2), (dim_unit1, dim_unit2), (coord_unit1, coord_unit2) = variants.get(variant)
    array1 = np.linspace(0, 1, 2 * 3).reshape(2, 3).astype(dtype) * data_unit1
    x1 = np.arange(2) * dim_unit1
    y1 = np.arange(3) * dim_unit1
    u1 = np.linspace(10, 20, 2) * coord_unit1
    v1 = np.linspace(10, 20, 3) * coord_unit1
    array2 = np.linspace(1, 2, 2 * 4).reshape(2, 4).astype(dtype) * data_unit2
    x2 = np.arange(2, 4) * dim_unit2
    z2 = np.arange(4) * dim_unit1
    u2 = np.linspace(20, 30, 2) * coord_unit2
    w2 = np.linspace(10, 20, 4) * coord_unit1
    array3 = np.linspace(0, 2, 3 * 4).reshape(3, 4).astype(dtype) * data_unit2
    y3 = np.arange(3, 6) * dim_unit2
    z3 = np.arange(4, 8) * dim_unit2
    v3 = np.linspace(10, 20, 3) * coord_unit2
    w3 = np.linspace(10, 20, 4) * coord_unit2
    arr1 = xr.DataArray(name='a', data=array1, coords={'x': x1, 'y': y1, 'u': ('x', u1), 'v': ('y', v1)}, dims=('x', 'y'))
    arr2 = xr.DataArray(name='a', data=array2, coords={'x': x2, 'z': z2, 'u': ('x', u2), 'w': ('z', w2)}, dims=('x', 'z'))
    arr3 = xr.DataArray(name='a', data=array3, coords={'y': y3, 'z': z3, 'v': ('y', v3), 'w': ('z', w3)}, dims=('y', 'z'))
    if error is not None:
        with pytest.raises(error):
            xr.merge([arr1, arr2, arr3])
        return
    units = {'a': data_unit1, 'u': coord_unit1, 'v': coord_unit1, 'w': coord_unit1, 'x': dim_unit1, 'y': dim_unit1, 'z': dim_unit1}
    convert_and_strip = lambda arr: strip_units(convert_units(arr, units))
    expected = attach_units(xr.merge([convert_and_strip(arr1), convert_and_strip(arr2), convert_and_strip(arr3)]), units)
    actual = xr.merge([arr1, arr2, arr3])
    assert_units_equal(expected, actual)
    assert_allclose(expected, actual)