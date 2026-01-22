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
def test_units_in_slice_line_plot_labels_sel(self, coord_unit, coord_attrs):
    arr = xr.DataArray(name='var_a', data=np.array([[1, 2], [3, 4]]), coords=dict(a=('a', np.array([5, 6]) * coord_unit, coord_attrs), b=('b', np.array([7, 8]) * coord_unit, coord_attrs)), dims=('a', 'b'))
    arr.sel(a=5).plot(marker='o')
    assert plt.gca().get_title() == 'a = 5 [meter]'