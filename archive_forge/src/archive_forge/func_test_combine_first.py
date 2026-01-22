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
@pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='same_unit')))
@pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units"))))
def test_combine_first(self, variant, unit, error, dtype):
    variants = {'data': (unit_registry.m, unit, 1, 1), 'dims': (1, 1, unit_registry.m, unit)}
    data_unit, other_data_unit, dims_unit, other_dims_unit = variants.get(variant)
    array1 = np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype) * data_unit
    array2 = np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype) * data_unit
    x = np.arange(len(array1)) * dims_unit
    ds = xr.Dataset(data_vars={'a': ('x', array1), 'b': ('x', array2)}, coords={'x': x})
    units = extract_units(ds)
    other_array1 = np.ones_like(array1) * other_data_unit
    other_array2 = np.full_like(array2, fill_value=-1) * other_data_unit
    other_x = (np.arange(array1.shape[0]) + 5) * other_dims_unit
    other = xr.Dataset(data_vars={'a': ('x', other_array1), 'b': ('x', other_array2)}, coords={'x': other_x})
    if error is not None:
        with pytest.raises(error):
            ds.combine_first(other)
        return
    expected = attach_units(strip_units(ds).combine_first(strip_units(convert_units(other, units))), units)
    actual = ds.combine_first(other)
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)