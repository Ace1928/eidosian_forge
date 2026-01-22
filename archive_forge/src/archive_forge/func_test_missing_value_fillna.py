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
@pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
def test_missing_value_fillna(self, unit, error):
    value = 10
    array = np.array([[1.4, 2.3, np.nan, 7.2], [np.nan, 9.7, np.nan, np.nan], [2.1, np.nan, np.nan, 4.6], [9.9, np.nan, 7.2, 9.1]]) * unit_registry.m
    variable = xr.Variable(('x', 'y'), array)
    fill_value = value * unit
    if error is not None:
        with pytest.raises(error):
            variable.fillna(value=fill_value)
        return
    expected = attach_units(strip_units(variable).fillna(value=fill_value.to(unit_registry.m).magnitude), extract_units(variable))
    actual = variable.fillna(value=fill_value)
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)