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
def test_aggregate_complex(self):
    variable = xr.Variable('x', [1, 2j, np.nan] * unit_registry.m)
    expected = xr.Variable((), (0.5 + 1j) * unit_registry.m)
    actual = variable.mean()
    assert_units_equal(expected, actual)
    assert_allclose(expected, actual)