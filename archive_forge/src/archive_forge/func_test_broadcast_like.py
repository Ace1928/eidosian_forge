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
@pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.s, id='incompatible_unit'), pytest.param(unit_registry.cm, id='compatible_unit'), pytest.param(unit_registry.m, id='identical_unit')))
@pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units"))))
def test_broadcast_like(self, variant, unit, dtype):
    variants = {'data': ((unit_registry.m, unit), (1, 1)), 'dims': ((1, 1), (unit_registry.m, unit))}
    (data_unit1, data_unit2), (dim_unit1, dim_unit2) = variants.get(variant)
    array1 = np.linspace(1, 2, 2 * 1).reshape(2, 1).astype(dtype) * data_unit1
    array2 = np.linspace(0, 1, 2 * 3).reshape(2, 3).astype(dtype) * data_unit2
    x1 = np.arange(2) * dim_unit1
    x2 = np.arange(2) * dim_unit2
    y1 = np.array([0]) * dim_unit1
    y2 = np.arange(3) * dim_unit2
    ds1 = xr.Dataset(data_vars={'a': (('x', 'y'), array1)}, coords={'x': x1, 'y': y1})
    ds2 = xr.Dataset(data_vars={'a': (('x', 'y'), array2)}, coords={'x': x2, 'y': y2})
    expected = attach_units(strip_units(ds1).broadcast_like(strip_units(ds2)), extract_units(ds1))
    actual = ds1.broadcast_like(ds2)
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)