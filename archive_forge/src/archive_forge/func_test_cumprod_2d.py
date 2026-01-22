from __future__ import annotations
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy import array, nan
from xarray import DataArray, Dataset, cftime_range, concat
from xarray.core import dtypes, duck_array_ops
from xarray.core.duck_array_ops import (
from xarray.namedarray.pycompat import array_type
from xarray.testing import assert_allclose, assert_equal, assert_identical
from xarray.tests import (
def test_cumprod_2d():
    inputs = np.array([[1, 2], [3, 4]])
    expected = np.array([[1, 2], [3, 2 * 3 * 4]])
    actual = duck_array_ops.cumprod(inputs)
    assert_array_equal(expected, actual)
    actual = duck_array_ops.cumprod(inputs, axis=(0, 1))
    assert_array_equal(expected, actual)
    actual = duck_array_ops.cumprod(inputs, axis=())
    assert_array_equal(inputs, actual)