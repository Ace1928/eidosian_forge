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
@pytest.mark.skip(reason="indexes don't support units")
@pytest.mark.parametrize('raw_values', (pytest.param(10, id='single_value'), pytest.param([10, 5, 13], id='list_of_values'), pytest.param(np.array([9, 3, 7, 12]), id='array_of_values')))
@pytest.mark.parametrize('unit,error', (pytest.param(1, KeyError, id='no_units'), pytest.param(unit_registry.dimensionless, KeyError, id='dimensionless'), pytest.param(unit_registry.degree, KeyError, id='incompatible_unit'), pytest.param(unit_registry.mm, KeyError, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
def test_drop_sel(self, raw_values, unit, error, dtype):
    array1 = np.linspace(5, 10, 20).astype(dtype) * unit_registry.degK
    array2 = np.linspace(0, 5, 20).astype(dtype) * unit_registry.Pa
    x = np.arange(len(array1)) * unit_registry.m
    ds = xr.Dataset(data_vars={'a': xr.DataArray(data=array1, dims='x'), 'b': xr.DataArray(data=array2, dims='x')}, coords={'x': x})
    values = raw_values * unit
    if error is not None:
        with pytest.raises(error):
            ds.drop_sel(x=values)
        return
    expected = attach_units(strip_units(ds).drop_sel(x=strip_units(convert_units(values, {None: unit_registry.m}))), extract_units(ds))
    actual = ds.drop_sel(x=values)
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)