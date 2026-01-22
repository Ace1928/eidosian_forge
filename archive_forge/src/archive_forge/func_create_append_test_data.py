from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def create_append_test_data(seed=None) -> tuple[Dataset, Dataset, Dataset]:
    rs = np.random.RandomState(seed)
    lat = [2, 1, 0]
    lon = [0, 1, 2]
    nt1 = 3
    nt2 = 2
    time1 = pd.date_range('2000-01-01', periods=nt1)
    time2 = pd.date_range('2000-02-01', periods=nt2)
    string_var = np.array(['a', 'bc', 'def'], dtype=object)
    string_var_to_append = np.array(['asdf', 'asdfg'], dtype=object)
    string_var_fixed_length = np.array(['aa', 'bb', 'cc'], dtype='|S2')
    string_var_fixed_length_to_append = np.array(['dd', 'ee'], dtype='|S2')
    unicode_var = np.array(['áó', 'áó', 'áó'])
    datetime_var = np.array(['2019-01-01', '2019-01-02', '2019-01-03'], dtype='datetime64[s]')
    datetime_var_to_append = np.array(['2019-01-04', '2019-01-05'], dtype='datetime64[s]')
    bool_var = np.array([True, False, True], dtype=bool)
    bool_var_to_append = np.array([False, True], dtype=bool)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Converting non-nanosecond')
        ds = xr.Dataset(data_vars={'da': xr.DataArray(rs.rand(3, 3, nt1), coords=[lat, lon, time1], dims=['lat', 'lon', 'time']), 'string_var': ('time', string_var), 'string_var_fixed_length': ('time', string_var_fixed_length), 'unicode_var': ('time', unicode_var), 'datetime_var': ('time', datetime_var), 'bool_var': ('time', bool_var)})
        ds_to_append = xr.Dataset(data_vars={'da': xr.DataArray(rs.rand(3, 3, nt2), coords=[lat, lon, time2], dims=['lat', 'lon', 'time']), 'string_var': ('time', string_var_to_append), 'string_var_fixed_length': ('time', string_var_fixed_length_to_append), 'unicode_var': ('time', unicode_var[:nt2]), 'datetime_var': ('time', datetime_var_to_append), 'bool_var': ('time', bool_var_to_append)})
        ds_with_new_var = xr.Dataset(data_vars={'new_var': xr.DataArray(rs.rand(3, 3, nt1 + nt2), coords=[lat, lon, time1.append(time2)], dims=['lat', 'lon', 'time'])})
    assert_writeable(ds)
    assert_writeable(ds_to_append)
    assert_writeable(ds_with_new_var)
    return (ds, ds_to_append, ds_with_new_var)