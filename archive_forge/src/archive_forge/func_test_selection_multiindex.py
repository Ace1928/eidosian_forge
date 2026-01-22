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
def test_selection_multiindex(self) -> None:
    midx = pd.MultiIndex.from_product([['a', 'b'], [1, 2], [-1, -2]], names=('one', 'two', 'three'))
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
    mdata = Dataset(data_vars={'var': ('x', range(8))}, coords=midx_coords)

    def test_sel(lab_indexer, pos_indexer, replaced_idx=False, renamed_dim=None) -> None:
        ds = mdata.sel(x=lab_indexer)
        expected_ds = mdata.isel(x=pos_indexer)
        if not replaced_idx:
            assert_identical(ds, expected_ds)
        else:
            if renamed_dim:
                assert ds['var'].dims[0] == renamed_dim
                ds = ds.rename({renamed_dim: 'x'})
            assert_identical(ds['var'].variable, expected_ds['var'].variable)
            assert not ds['x'].equals(expected_ds['x'])
    test_sel(('a', 1, -1), 0)
    test_sel(('b', 2, -2), -1)
    test_sel(('a', 1), [0, 1], replaced_idx=True, renamed_dim='three')
    test_sel(('a',), range(4), replaced_idx=True)
    test_sel('a', range(4), replaced_idx=True)
    test_sel([('a', 1, -1), ('b', 2, -2)], [0, 7])
    test_sel(slice('a', 'b'), range(8))
    test_sel(slice(('a', 1), ('b', 1)), range(6))
    test_sel({'one': 'a', 'two': 1, 'three': -1}, 0)
    test_sel({'one': 'a', 'two': 1}, [0, 1], replaced_idx=True, renamed_dim='three')
    test_sel({'one': 'a'}, range(4), replaced_idx=True)
    assert_identical(mdata.loc[{'x': {'one': 'a'}}], mdata.sel(x={'one': 'a'}))
    assert_identical(mdata.loc[{'x': 'a'}], mdata.sel(x='a'))
    assert_identical(mdata.loc[{'x': ('a', 1)}], mdata.sel(x=('a', 1)))
    assert_identical(mdata.loc[{'x': ('a', 1, -1)}], mdata.sel(x=('a', 1, -1)))
    assert_identical(mdata.sel(x={'one': 'a', 'two': 1}), mdata.sel(one='a', two=1))