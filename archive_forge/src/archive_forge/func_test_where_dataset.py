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
@pytest.mark.parametrize('fill_value', (np.nan, 10.2))
def test_where_dataset(fill_value, unit, error, dtype):
    array1 = np.linspace(0, 5, 10).astype(dtype) * unit_registry.m
    array2 = np.linspace(-5, 0, 10).astype(dtype) * unit_registry.m
    ds = xr.Dataset(data_vars={'a': ('x', array1), 'b': ('x', array2)})
    cond = array1 < 2 * unit_registry.m
    fill_value = fill_value * unit
    if error is not None and (not (np.isnan(fill_value) and (not isinstance(fill_value, Quantity)))):
        with pytest.raises(error):
            xr.where(cond, ds, fill_value)
        return
    expected = attach_units(xr.where(cond, strip_units(ds), strip_units(convert_units(fill_value, {None: unit_registry.m}))), extract_units(ds))
    actual = xr.where(cond, ds, fill_value)
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)