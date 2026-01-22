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
@pytest.mark.parametrize('func', (method('item', 5), method('searchsorted', 5)), ids=repr)
@pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
def test_raw_numpy_methods(self, func, unit, error, dtype):
    array = np.linspace(0, 1, 10).astype(dtype) * unit_registry.m
    variable = xr.Variable('x', array)
    args = [item * unit if isinstance(item, (int, float, list)) and func.name != 'item' else item for item in func.args]
    kwargs = {key: value * unit if isinstance(value, (int, float, list)) and func.name != 'item' else value for key, value in func.kwargs.items()}
    if error is not None and func.name != 'item':
        with pytest.raises(error):
            func(variable, *args, **kwargs)
        return
    converted_args = [strip_units(convert_units(item, {None: unit_registry.m})) if func.name != 'item' else item for item in args]
    converted_kwargs = {key: strip_units(convert_units(value, {None: unit_registry.m})) if func.name != 'item' else value for key, value in kwargs.items()}
    units = extract_units(func(array, *args, **kwargs))
    expected = attach_units(func(strip_units(variable), *converted_args, **converted_kwargs), units)
    actual = func(variable, *args, **kwargs)
    assert_units_equal(expected, actual)
    assert_duckarray_allclose(expected, actual)