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
@given(st.data(), dimension_names())
def test_given_fixed_dim_names(self, data, fixed_dim_names):
    var = data.draw(variables(dims=st.just(fixed_dim_names)))
    assert list(var.dims) == fixed_dim_names