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
def test_where_type_promotion(self):
    result = where([True, False], [1, 2], ['a', 'b'])
    assert_array_equal(result, np.array([1, 'b'], dtype=object))
    result = where([True, False], np.array([1, 2], np.float32), np.nan)
    assert result.dtype == np.float32
    assert_array_equal(result, np.array([1, np.nan], dtype=np.float32))