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
def test_merge_dataset(variant, unit, error, dtype):
    original_unit = unit_registry.m
    variants = {'data': ((original_unit, unit), (1, 1), (1, 1)), 'dims': ((1, 1), (original_unit, unit), (1, 1)), 'coords': ((1, 1), (1, 1), (original_unit, unit))}
    (data_unit1, data_unit2), (dim_unit1, dim_unit2), (coord_unit1, coord_unit2) = variants.get(variant)
    array1 = np.zeros(shape=(2, 3), dtype=dtype) * data_unit1
    array2 = np.zeros(shape=(2, 3), dtype=dtype) * data_unit1
    x = np.arange(11, 14) * dim_unit1
    y = np.arange(2) * dim_unit1
    u = np.arange(3) * coord_unit1
    ds1 = xr.Dataset(data_vars={'a': (('y', 'x'), array1), 'b': (('y', 'x'), array2)}, coords={'x': x, 'y': y, 'u': ('x', u)})
    ds2 = xr.Dataset(data_vars={'a': (('y', 'x'), np.ones_like(array1) * data_unit2), 'b': (('y', 'x'), np.ones_like(array2) * data_unit2)}, coords={'x': np.arange(3) * dim_unit2, 'y': np.arange(2, 4) * dim_unit2, 'u': ('x', np.arange(-3, 0) * coord_unit2)})
    ds3 = xr.Dataset(data_vars={'a': (('y', 'x'), np.full_like(array1, np.nan) * data_unit2), 'b': (('y', 'x'), np.full_like(array2, np.nan) * data_unit2)}, coords={'x': np.arange(3, 6) * dim_unit2, 'y': np.arange(4, 6) * dim_unit2, 'u': ('x', np.arange(3, 6) * coord_unit2)})
    func = function(xr.merge)
    if error is not None:
        with pytest.raises(error):
            func([ds1, ds2, ds3])
        return
    units = extract_units(ds1)
    convert_and_strip = lambda ds: strip_units(convert_units(ds, units))
    expected = attach_units(func([convert_and_strip(ds1), convert_and_strip(ds2), convert_and_strip(ds3)]), units)
    actual = func([ds1, ds2, ds3])
    assert_units_equal(expected, actual)
    assert_allclose(expected, actual)