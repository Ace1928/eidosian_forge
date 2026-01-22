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
def test_map_blocks_dask_args():
    da1 = xr.DataArray(np.ones((10, 20)), dims=['x', 'y'], coords={'x': np.arange(10), 'y': np.arange(20)}).chunk({'x': 5, 'y': 4})

    def sumda(da1, da2):
        assert da1.shape == da2.shape
        return da1 + da2
    da2 = da1 + 1
    with raise_if_dask_computes():
        mapped = xr.map_blocks(sumda, da1, args=[da2])
    xr.testing.assert_equal(da1 + da2, mapped)
    da2 = (da1 + 1).isel(x=1, drop=True)
    with raise_if_dask_computes():
        mapped = xr.map_blocks(operator.add, da1, args=[da2])
    xr.testing.assert_equal(da1 + da2, mapped)
    da2 = (da1 + 1).isel(x=1, drop=True).rename({'y': 'k'})
    with raise_if_dask_computes():
        mapped = xr.map_blocks(operator.add, da1, args=[da2])
    xr.testing.assert_equal(da1 + da2, mapped)
    with pytest.raises(ValueError, match="Chunk sizes along dimension 'x'"):
        xr.map_blocks(operator.add, da1, args=[da1.chunk({'x': 1})])
    with pytest.raises(ValueError, match='cannot align.*index.*are not equal'):
        xr.map_blocks(operator.add, da1, args=[da1.reindex(x=np.arange(20))])
    da1 = da1.chunk({'x': -1})
    da2 = da1 + 1
    with raise_if_dask_computes():
        mapped = xr.map_blocks(lambda a, b: (a + b).sum('x'), da1, args=[da2])
    xr.testing.assert_equal((da1 + da2).sum('x'), mapped)
    da1 = da1.chunk({'x': -1})
    da2 = da1 + 1
    with raise_if_dask_computes():
        mapped = xr.map_blocks(lambda a, b: (a + b).sum('x'), da1, args=[da2], template=da1.sum('x'))
    xr.testing.assert_equal((da1 + da2).sum('x'), mapped)
    with pytest.raises(ValueError, match='Provided template has no dask arrays'):
        xr.map_blocks(lambda a, b: (a + b).sum('x'), da1, args=[da2], template=da1.sum('x').compute())