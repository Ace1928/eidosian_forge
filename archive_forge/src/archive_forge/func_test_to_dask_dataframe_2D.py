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
@pytest.mark.xfail(reason='Currently pandas with pyarrow installed will return a `string[pyarrow]` type, which causes the `y` column to have a different type depending on whether pyarrow is installed')
def test_to_dask_dataframe_2D(self):
    w = np.random.randn(2, 3)
    ds = Dataset({'w': (('x', 'y'), da.from_array(w, chunks=(1, 2)))})
    ds['x'] = ('x', np.array([0, 1], np.int64))
    ds['y'] = ('y', list('abc'))
    exp_index = pd.MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], ['a', 'b', 'c', 'a', 'b', 'c']], names=['x', 'y'])
    expected = pd.DataFrame({'w': w.reshape(-1)}, index=exp_index)
    expected = expected.reset_index(drop=False)
    actual = ds.to_dask_dataframe(set_index=False)
    assert isinstance(actual, dd.DataFrame)
    assert_frame_equal(actual.compute(), expected)