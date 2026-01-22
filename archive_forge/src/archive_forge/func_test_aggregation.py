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
@pytest.mark.parametrize('func', (method('all'), method('any'), method('argmax', dim='x'), method('argmin', dim='x'), method('max'), method('min'), method('mean'), method('median'), method('sum'), method('prod'), method('std'), method('var'), method('cumsum'), method('cumprod')), ids=repr)
def test_aggregation(self, func, dtype):
    unit_a, unit_b = (unit_registry.Pa, unit_registry.degK) if func.name != 'cumprod' else (unit_registry.dimensionless, unit_registry.dimensionless)
    a = np.linspace(0, 1, 10).astype(dtype) * unit_a
    b = np.linspace(-1, 0, 10).astype(dtype) * unit_b
    ds = xr.Dataset({'a': ('x', a), 'b': ('x', b)})
    if 'dim' in func.kwargs:
        numpy_kwargs = func.kwargs.copy()
        dim = numpy_kwargs.pop('dim')
        axis_a = ds.a.get_axis_num(dim)
        axis_b = ds.b.get_axis_num(dim)
        numpy_kwargs_a = numpy_kwargs.copy()
        numpy_kwargs_a['axis'] = axis_a
        numpy_kwargs_b = numpy_kwargs.copy()
        numpy_kwargs_b['axis'] = axis_b
    else:
        numpy_kwargs_a = {}
        numpy_kwargs_b = {}
    units_a = array_extract_units(func(a, **numpy_kwargs_a))
    units_b = array_extract_units(func(b, **numpy_kwargs_b))
    units = {'a': units_a, 'b': units_b}
    actual = func(ds)
    expected = attach_units(func(strip_units(ds)), units)
    assert_units_equal(expected, actual)
    assert_allclose(expected, actual)