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
@given(st.data(), supported_dtypes())
def test_given_fixed_dtype(self, data, dtype):
    var = data.draw(variables(dtype=st.just(dtype)))
    assert var.dtype == dtype