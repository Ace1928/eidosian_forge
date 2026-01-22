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
def test_broadcast_equals(self, unit, dtype):
    left_array1 = np.ones(shape=(2, 3), dtype=dtype) * unit_registry.m
    left_array2 = np.zeros(shape=(3, 6), dtype=dtype) * unit_registry.m
    right_array1 = np.ones(shape=(2,)) * unit
    right_array2 = np.zeros(shape=(3,)) * unit
    left = xr.Dataset({'a': (('x', 'y'), left_array1), 'b': (('y', 'z'), left_array2)})
    right = xr.Dataset({'a': ('x', right_array1), 'b': ('y', right_array2)})
    units = merge_mappings(extract_units(left), {} if is_compatible(left_array1, unit) else {'a': None, 'b': None})
    expected = is_compatible(left_array1, unit) and strip_units(left).broadcast_equals(strip_units(convert_units(right, units)))
    actual = left.broadcast_equals(right)
    assert expected == actual