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
def test_recursive_token():
    """Test that tokenization is invoked recursively, and doesn't just rely on the
    output of str()
    """
    a = np.ones(10000)
    b = np.ones(10000)
    b[5000] = 2
    assert str(a) == str(b)
    assert dask.base.tokenize(a) != dask.base.tokenize(b)
    da_a = DataArray(a)
    da_b = DataArray(b)
    assert dask.base.tokenize(da_a) != dask.base.tokenize(da_b)
    ds_a = da_a.to_dataset(name='x')
    ds_b = da_b.to_dataset(name='x')
    assert dask.base.tokenize(ds_a) != dask.base.tokenize(ds_b)
    da_a = DataArray(a, dims=['x'], coords={'x': a})
    da_b = DataArray(a, dims=['x'], coords={'x': b})
    assert dask.base.tokenize(da_a) != dask.base.tokenize(da_b)