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
def test_empty_axis_dtype():
    ds = Dataset()
    ds['pos'] = [1, 2, 3]
    ds['data'] = (('pos', 'time'), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ds['var'] = ('pos', [2, 3, 4])
    assert_identical(ds.mean(dim='time')['var'], ds['var'])
    assert_identical(ds.max(dim='time')['var'], ds['var'])
    assert_identical(ds.min(dim='time')['var'], ds['var'])
    assert_identical(ds.sum(dim='time')['var'], ds['var'])