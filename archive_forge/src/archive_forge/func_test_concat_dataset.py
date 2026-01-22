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
def test_concat_dataset(variant, unit, error, dtype):
    original_unit = unit_registry.m
    variants = {'data': ((original_unit, unit), (1, 1), (1, 1)), 'dims': ((1, 1), (original_unit, unit), (1, 1)), 'coords': ((1, 1), (1, 1), (original_unit, unit))}
    (data_unit1, data_unit2), (dim_unit1, dim_unit2), (coord_unit1, coord_unit2) = variants.get(variant)
    array1 = np.linspace(0, 5, 10).astype(dtype) * data_unit1
    array2 = np.linspace(-5, 0, 5).astype(dtype) * data_unit2
    x1 = np.arange(5, 15) * dim_unit1
    x2 = np.arange(5) * dim_unit2
    u1 = np.linspace(1, 2, 10).astype(dtype) * coord_unit1
    u2 = np.linspace(0, 1, 5).astype(dtype) * coord_unit2
    ds1 = xr.Dataset(data_vars={'a': ('x', array1)}, coords={'x': x1, 'u': ('x', u1)})
    ds2 = xr.Dataset(data_vars={'a': ('x', array2)}, coords={'x': x2, 'u': ('x', u2)})
    if error is not None:
        with pytest.raises(error):
            xr.concat([ds1, ds2], dim='x')
        return
    units = extract_units(ds1)
    expected = attach_units(xr.concat([strip_units(ds1), strip_units(convert_units(ds2, units))], dim='x'), units)
    actual = xr.concat([ds1, ds2], dim='x')
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)