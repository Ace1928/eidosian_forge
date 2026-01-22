from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
def test_to_dask_dataframe_dim_order(self):
    values = np.array([[1, 2], [3, 4]], dtype=np.int64)
    ds = Dataset({'w': (('x', 'y'), values)}).chunk(1)
    expected = ds['w'].to_series().reset_index()
    actual = ds.to_dask_dataframe(dim_order=['x', 'y'])
    assert isinstance(actual, dd.DataFrame)
    assert_frame_equal(expected, actual.compute())
    expected = ds['w'].T.to_series().reset_index()
    actual = ds.to_dask_dataframe(dim_order=['y', 'x'])
    assert isinstance(actual, dd.DataFrame)
    assert_frame_equal(expected, actual.compute())
    with pytest.raises(ValueError, match='does not match the set of dimensions'):
        ds.to_dask_dataframe(dim_order=['x'])