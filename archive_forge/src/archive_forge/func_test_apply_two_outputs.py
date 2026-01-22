from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
def test_apply_two_outputs() -> None:
    array = np.arange(5)
    variable = xr.Variable('x', array)
    data_array = xr.DataArray(variable, [('x', -array)])
    dataset = xr.Dataset({'y': variable}, {'x': -array})

    def twice(obj):

        def func(x):
            return (x, x)
        return apply_ufunc(func, obj, output_core_dims=[[], []])
    out0, out1 = twice(array)
    assert_identical(out0, array)
    assert_identical(out1, array)
    out0, out1 = twice(variable)
    assert_identical(out0, variable)
    assert_identical(out1, variable)
    out0, out1 = twice(data_array)
    assert_identical(out0, data_array)
    assert_identical(out1, data_array)
    out0, out1 = twice(dataset)
    assert_identical(out0, dataset)
    assert_identical(out1, dataset)
    out0, out1 = twice(data_array.groupby('x'))
    assert_identical(out0, data_array)
    assert_identical(out1, data_array)
    out0, out1 = twice(dataset.groupby('x'))
    assert_identical(out0, dataset)
    assert_identical(out1, dataset)