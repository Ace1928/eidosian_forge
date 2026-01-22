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
@pytest.mark.xfail(reason='Currently pandas with pyarrow installed will return a `string[pyarrow]` type, which causes the index to have a different type depending on whether pyarrow is installed')
def test_to_dask_dataframe_not_daskarray(self):
    x = np.random.randn(10)
    y = np.arange(10, dtype='uint8')
    t = list('abcdefghij')
    ds = Dataset({'a': ('t', x), 'b': ('t', y), 't': ('t', t)})
    expected = pd.DataFrame({'a': x, 'b': y}, index=pd.Index(t, name='t'))
    actual = ds.to_dask_dataframe(set_index=True)
    assert isinstance(actual, dd.DataFrame)
    assert_frame_equal(expected, actual.compute())