import numpy as np
import numpy.testing as npt
import pytest
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.extra.array_api import make_strategies_namespace
from xarray.core.variable import Variable
from xarray.testing.strategies import (
from xarray.tests import requires_numpy_array_api
@given(st.data(), npst.arrays(shape=npst.array_shapes(), dtype=supported_dtypes()))
def test_given_fixed_data_dims_and_dtype(self, data, arr):

    def fixed_array_strategy_fn(*, shape=None, dtype=None):
        """The fact this ignores shape and dtype is only okay because compatible shape & dtype will be passed separately."""
        return st.just(arr)
    dim_names = data.draw(dimension_names(min_dims=arr.ndim, max_dims=arr.ndim))
    dim_sizes = {name: size for name, size in zip(dim_names, arr.shape)}
    var = data.draw(variables(array_strategy_fn=fixed_array_strategy_fn, dims=st.just(dim_sizes), dtype=st.just(arr.dtype)))
    npt.assert_equal(var.data, arr)
    assert var.dtype == arr.dtype