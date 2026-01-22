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
def test_map_blocks_error(map_da, map_ds):

    def bad_func(darray):
        return (darray * darray.x + 5 * darray.y)[:1, :1]
    with pytest.raises(ValueError, match="Received dimension 'x' of length 1"):
        xr.map_blocks(bad_func, map_da).compute()

    def returns_numpy(darray):
        return (darray * darray.x + 5 * darray.y).values
    with pytest.raises(TypeError, match='Function must return an xarray DataArray'):
        xr.map_blocks(returns_numpy, map_da)
    with pytest.raises(TypeError, match='args must be'):
        xr.map_blocks(operator.add, map_da, args=10)
    with pytest.raises(TypeError, match='kwargs must be'):
        xr.map_blocks(operator.add, map_da, args=[10], kwargs=[20])

    def really_bad_func(darray):
        raise ValueError("couldn't do anything.")
    with pytest.raises(Exception, match='Cannot infer'):
        xr.map_blocks(really_bad_func, map_da)
    ds_copy = map_ds.copy()
    ds_copy['cxy'] = ds_copy.cxy.chunk({'y': 10})
    with pytest.raises(ValueError, match='inconsistent chunks'):
        xr.map_blocks(bad_func, ds_copy)
    with pytest.raises(TypeError, match='Cannot pass dask collections'):
        xr.map_blocks(bad_func, map_da, kwargs=dict(a=map_da.chunk()))