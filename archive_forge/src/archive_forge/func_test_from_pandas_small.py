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
def test_from_pandas_small():
    df = pd.DataFrame({'x': [1, 2, 3]})
    for i in [1, 2, 30]:
        a = dd.from_pandas(df, i)
        assert len(a.compute()) == 3
        assert a.divisions[0] == 0
        assert a.divisions[-1] == 2
        a = dd.from_pandas(df, chunksize=i)
        assert len(a.compute()) == 3
        assert a.divisions[0] == 0
        assert a.divisions[-1] == 2
    for sort in [True, False]:
        for i in [0, 2]:
            df = pd.DataFrame({'x': [0] * i})
            ddf = dd.from_pandas(df, npartitions=5, sort=sort)
            assert_eq(df, ddf)
            s = pd.Series([0] * i, name='x', dtype=int)
            ds = dd.from_pandas(s, npartitions=5, sort=sort)
            assert_eq(s, ds)