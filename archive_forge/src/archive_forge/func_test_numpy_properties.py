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
@pytest.mark.parametrize('property', ('imag', 'real'))
def test_numpy_properties(self, property, dtype):
    a = np.linspace(0, 1, 10) * unit_registry.Pa
    b = np.linspace(-1, 0, 15) * unit_registry.degK
    ds = xr.Dataset({'a': ('x', a), 'b': ('y', b)})
    units = extract_units(ds)
    actual = getattr(ds, property)
    expected = attach_units(getattr(strip_units(ds), property), units)
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)