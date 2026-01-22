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
def test_broadcast_compat_data_2d() -> None:
    data = np.arange(12).reshape(3, 4)
    var = xr.Variable(['x', 'y'], data)
    assert_identical(data, broadcast_compat_data(var, ('x', 'y'), ()))
    assert_identical(data, broadcast_compat_data(var, ('x',), ('y',)))
    assert_identical(data, broadcast_compat_data(var, (), ('x', 'y')))
    assert_identical(data.T, broadcast_compat_data(var, ('y', 'x'), ()))
    assert_identical(data.T, broadcast_compat_data(var, ('y',), ('x',)))
    assert_identical(data, broadcast_compat_data(var, ('w', 'x'), ('y',)))
    assert_identical(data, broadcast_compat_data(var, ('w',), ('x', 'y')))
    assert_identical(data.T, broadcast_compat_data(var, ('w',), ('y', 'x')))
    assert_identical(data[:, :, None], broadcast_compat_data(var, ('w', 'x', 'y', 'z'), ()))
    assert_identical(data[None, :, :].T, broadcast_compat_data(var, ('w', 'y', 'x', 'z'), ()))