from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import dask
import dask.array as da
import dask.dataframe as dd
from dask import config
from dask.blockwise import Blockwise
from dask.dataframe._compat import PANDAS_GE_200, tm
from dask.dataframe.io.io import _meta_from_array
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import assert_eq, get_string_dtype, pyarrow_strings_enabled
from dask.delayed import Delayed, delayed
from dask.utils_test import hlg_layer_topological
def test_from_map_meta():
    string_dtype = get_string_dtype()

    def func(x, s=0):
        df = pd.DataFrame({'x': [x] * s})
        return df
    iterable = ['A', 'B']
    expect = pd.DataFrame({'x': ['A', 'A', 'B', 'B']}, index=[0, 1, 0, 1])
    meta = pd.DataFrame({'x': pd.Series(['A'], dtype=string_dtype)}).iloc[:0]
    ddf = dd.from_map(func, iterable, meta=meta, s=2)
    assert_eq(ddf._meta, meta)
    assert_eq(ddf, expect)
    meta = pd.DataFrame({'a': pd.Series(['A'], dtype=string_dtype)}).iloc[:0]
    ddf = dd.from_map(func, iterable, meta=meta, s=2)
    assert_eq(ddf._meta, meta)
    if not DASK_EXPR_ENABLED:
        with pytest.raises(ValueError, match='The columns in the computed data'):
            assert_eq(ddf.compute(), expect)
    ddf = dd.from_map(func, iterable, meta=meta, enforce_metadata=False, s=2)
    assert_eq(ddf._meta, meta)
    assert_eq(ddf.compute(), expect)