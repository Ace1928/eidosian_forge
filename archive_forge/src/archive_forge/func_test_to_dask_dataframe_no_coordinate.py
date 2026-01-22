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
def test_to_dask_dataframe_no_coordinate(self):
    x = da.from_array(np.random.randn(10), chunks=4)
    ds = Dataset({'x': ('dim_0', x)})
    expected = ds.compute().to_dataframe().reset_index()
    actual = ds.to_dask_dataframe()
    assert isinstance(actual, dd.DataFrame)
    assert_frame_equal(expected, actual.compute())
    expected = ds.compute().to_dataframe()
    actual = ds.to_dask_dataframe(set_index=True)
    assert isinstance(actual, dd.DataFrame)
    assert_frame_equal(expected, actual.compute())