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
def test_pad_unit_constant_value(self, unit, error, dtype):
    array = np.linspace(0, 5, 3 * 10).reshape(3, 10).astype(dtype) * unit_registry.m
    variable = xr.Variable(('x', 'y'), array)
    fill_value = -100 * unit
    func = method('pad', mode='constant', x=(2, 3), y=(1, 4))
    if error is not None:
        with pytest.raises(error):
            func(variable, constant_values=fill_value)
        return
    units = extract_units(variable)
    expected = attach_units(func(strip_units(variable), constant_values=strip_units(convert_units(fill_value, units))), units)
    actual = func(variable, constant_values=fill_value)
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)