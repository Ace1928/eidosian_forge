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
def test_apply_output_core_dimension() -> None:

    def stack_negative(obj):

        def func(x):
            return np.stack([x, -x], axis=-1)
        result = apply_ufunc(func, obj, output_core_dims=[['sign']])
        if isinstance(result, (xr.Dataset, xr.DataArray)):
            result.coords['sign'] = [1, -1]
        return result
    array = np.array([[1, 2], [3, 4]])
    variable = xr.Variable(['x', 'y'], array)
    data_array = xr.DataArray(variable, {'x': ['a', 'b'], 'y': [-1, -2]})
    dataset = xr.Dataset({'data': data_array})
    stacked_array = np.array([[[1, -1], [2, -2]], [[3, -3], [4, -4]]])
    stacked_variable = xr.Variable(['x', 'y', 'sign'], stacked_array)
    stacked_coords = {'x': ['a', 'b'], 'y': [-1, -2], 'sign': [1, -1]}
    stacked_data_array = xr.DataArray(stacked_variable, stacked_coords)
    stacked_dataset = xr.Dataset({'data': stacked_data_array})
    assert_identical(stacked_array, stack_negative(array))
    assert_identical(stacked_variable, stack_negative(variable))
    assert_identical(stacked_data_array, stack_negative(data_array))
    assert_identical(stacked_dataset, stack_negative(dataset))
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        assert_identical(stacked_data_array, stack_negative(data_array.groupby('x')))
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        assert_identical(stacked_dataset, stack_negative(dataset.groupby('x')))

    def original_and_stack_negative(obj):

        def func(x):
            return (x, np.stack([x, -x], axis=-1))
        result = apply_ufunc(func, obj, output_core_dims=[[], ['sign']])
        if isinstance(result[1], (xr.Dataset, xr.DataArray)):
            result[1].coords['sign'] = [1, -1]
        return result
    out0, out1 = original_and_stack_negative(array)
    assert_identical(array, out0)
    assert_identical(stacked_array, out1)
    out0, out1 = original_and_stack_negative(variable)
    assert_identical(variable, out0)
    assert_identical(stacked_variable, out1)
    out0, out1 = original_and_stack_negative(data_array)
    assert_identical(data_array, out0)
    assert_identical(stacked_data_array, out1)
    out0, out1 = original_and_stack_negative(dataset)
    assert_identical(dataset, out0)
    assert_identical(stacked_dataset, out1)
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        out0, out1 = original_and_stack_negative(data_array.groupby('x'))
    assert_identical(data_array, out0)
    assert_identical(stacked_data_array, out1)
    with pytest.warns(UserWarning, match='The `squeeze` kwarg'):
        out0, out1 = original_and_stack_negative(dataset.groupby('x'))
    assert_identical(dataset, out0)
    assert_identical(stacked_dataset, out1)