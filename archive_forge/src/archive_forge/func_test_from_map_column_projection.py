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
@pytest.mark.xfail(DASK_EXPR_ENABLED, reason='hashing not deterministic')
def test_from_map_column_projection():
    projected = []

    class MyFunc:

        def __init__(self, columns=None):
            self.columns = columns

        def project_columns(self, columns):
            return MyFunc(columns)

        def __call__(self, t):
            size = t[0] + 1
            x = t[1]
            df = pd.DataFrame({'A': [x] * size, 'B': [10] * size})
            if self.columns is None:
                return df
            projected.extend(self.columns)
            return df[self.columns]
    ddf = dd.from_map(MyFunc(), enumerate([0, 1, 2]), label='myfunc', enforce_metadata=True)
    expect = pd.DataFrame({'A': [0, 1, 1, 2, 2, 2], 'B': [10] * 6}, index=[0, 0, 1, 0, 1, 2])
    assert_eq(ddf['A'], expect['A'])
    assert set(projected) == {'A'}
    assert_eq(ddf, expect)