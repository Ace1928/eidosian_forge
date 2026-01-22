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
@requires_cftime
@requires_dask
def test_mean_over_non_time_dim_of_dataset_with_dask_backed_cftime_data():
    ds = Dataset({'var1': (('time',), cftime_range('2021-10-31', periods=10, freq='D')), 'var2': (('x',), list(range(10)))})
    expected = ds.mean('x')
    result = ds.chunk({}).mean('x')
    assert_equal(result, expected)