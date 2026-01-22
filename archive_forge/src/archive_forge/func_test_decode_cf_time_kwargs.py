from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from xarray import (
from xarray.backends.common import WritableCFDataStore
from xarray.backends.memory import InMemoryDataStore
from xarray.conventions import decode_cf
from xarray.testing import assert_identical
from xarray.tests import (
from xarray.tests.test_backends import CFEncodedBase
def test_decode_cf_time_kwargs(self) -> None:
    ds = Dataset.from_dict({'coords': {'timedelta': {'data': np.array([1, 2, 3], dtype='int64'), 'dims': 'timedelta', 'attrs': {'units': 'days'}}, 'time': {'data': np.array([1, 2, 3], dtype='int64'), 'dims': 'time', 'attrs': {'units': 'days since 2000-01-01'}}}, 'dims': {'time': 3, 'timedelta': 3}, 'data_vars': {'a': {'dims': ('time', 'timedelta'), 'data': np.ones((3, 3))}}})
    dsc = conventions.decode_cf(ds)
    assert dsc.timedelta.dtype == np.dtype('m8[ns]')
    assert dsc.time.dtype == np.dtype('M8[ns]')
    dsc = conventions.decode_cf(ds, decode_times=False)
    assert dsc.timedelta.dtype == np.dtype('int64')
    assert dsc.time.dtype == np.dtype('int64')
    dsc = conventions.decode_cf(ds, decode_times=True, decode_timedelta=False)
    assert dsc.timedelta.dtype == np.dtype('int64')
    assert dsc.time.dtype == np.dtype('M8[ns]')
    dsc = conventions.decode_cf(ds, decode_times=False, decode_timedelta=True)
    assert dsc.timedelta.dtype == np.dtype('m8[ns]')
    assert dsc.time.dtype == np.dtype('int64')