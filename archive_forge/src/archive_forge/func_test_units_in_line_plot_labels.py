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
@pytest.mark.parametrize('coord_unit, coord_attrs', [(1, {'units': 'meter'}), pytest.param(unit_registry.m, {}, marks=pytest.mark.xfail(reason="indexes don't support units"))])
def test_units_in_line_plot_labels(self, coord_unit, coord_attrs):
    arr = np.linspace(1, 10, 3) * unit_registry.Pa
    coord_arr = np.linspace(1, 3, 3) * coord_unit
    x_coord = xr.DataArray(coord_arr, dims='x', attrs=coord_attrs)
    da = xr.DataArray(data=arr, dims='x', coords={'x': x_coord}, name='pressure')
    da.plot.line()
    ax = plt.gca()
    assert ax.get_ylabel() == 'pressure [pascal]'
    assert ax.get_xlabel() == 'x [meter]'