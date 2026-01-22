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
def test_concat_loads_variables(self):
    d1 = build_dask_array('d1')
    c1 = build_dask_array('c1')
    d2 = build_dask_array('d2')
    c2 = build_dask_array('c2')
    d3 = build_dask_array('d3')
    c3 = build_dask_array('c3')
    ds1 = Dataset(data_vars={'d': ('x', d1)}, coords={'c': ('x', c1)})
    ds2 = Dataset(data_vars={'d': ('x', d2)}, coords={'c': ('x', c2)})
    ds3 = Dataset(data_vars={'d': ('x', d3)}, coords={'c': ('x', c3)})
    assert kernel_call_count == 0
    out = xr.concat([ds1, ds2, ds3], dim='n', data_vars='different', coords='different')
    assert kernel_call_count == 6
    assert isinstance(out['d'].data, np.ndarray)
    assert isinstance(out['c'].data, np.ndarray)
    out = xr.concat([ds1, ds2, ds3], dim='n', data_vars='all', coords='all')
    assert kernel_call_count == 6
    assert isinstance(out['d'].data, dask.array.Array)
    assert isinstance(out['c'].data, dask.array.Array)
    out = xr.concat([ds1, ds2, ds3], dim='n', data_vars=['d'], coords=['c'])
    assert kernel_call_count == 6
    assert isinstance(out['d'].data, dask.array.Array)
    assert isinstance(out['c'].data, dask.array.Array)
    out = xr.concat([ds1, ds2, ds3], dim='n', data_vars=[], coords=[])
    assert kernel_call_count == 12
    assert isinstance(out['d'].data, np.ndarray)
    assert isinstance(out['c'].data, np.ndarray)
    out = xr.concat([ds1, ds2, ds3], dim='n', data_vars='different', coords='different', compat='identical')
    assert kernel_call_count == 18
    assert isinstance(out['d'].data, np.ndarray)
    assert isinstance(out['c'].data, np.ndarray)
    ds4 = Dataset(data_vars={'d': ('x', [2.0])}, coords={'c': ('x', [2.0])})
    out = xr.concat([ds1, ds2, ds4, ds3], dim='n', data_vars='different', coords='different')
    assert kernel_call_count == 22
    assert isinstance(out['d'].data, dask.array.Array)
    assert isinstance(out['c'].data, dask.array.Array)
    out.compute()
    assert kernel_call_count == 24
    assert ds1['d'].data is d1
    assert ds1['c'].data is c1
    assert ds2['d'].data is d2
    assert ds2['c'].data is c2
    assert ds3['d'].data is d3
    assert ds3['c'].data is c3
    out = xr.concat([ds1, ds1, ds1], dim='n', data_vars='different', coords='different')
    assert kernel_call_count == 24
    assert isinstance(out['d'].data, dask.array.Array)
    assert isinstance(out['c'].data, dask.array.Array)
    out = xr.concat([ds1, ds1, ds1], dim='n', data_vars=[], coords=[], compat='identical')
    assert kernel_call_count == 24
    assert isinstance(out['d'].data, dask.array.Array)
    assert isinstance(out['c'].data, dask.array.Array)
    out = xr.concat([ds1, ds2.compute(), ds3], dim='n', data_vars='all', coords='different', compat='identical')
    assert kernel_call_count == 28
    out = xr.concat([ds1, ds2.compute(), ds3], dim='n', data_vars='all', coords='all', compat='identical')
    assert kernel_call_count == 30
    assert ds1['d'].data is d1
    assert ds1['c'].data is c1
    assert ds2['d'].data is d2
    assert ds2['c'].data is c2
    assert ds3['d'].data is d3
    assert ds3['c'].data is c3