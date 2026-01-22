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
def test_from_pandas_series():
    n = 20
    s = pd.Series(np.random.randn(n), index=pd.date_range(start='20120101', periods=n))
    ds = dd.from_pandas(s, 3)
    assert len(ds.dask) == 3
    assert len(ds.divisions) == len(ds.dask) + 1
    assert isinstance(ds.divisions[0], type(s.index[0]))
    tm.assert_series_equal(s, ds.compute())
    ds = dd.from_pandas(s, chunksize=8)
    assert len(ds.dask) == 3
    assert len(ds.divisions) == len(ds.dask) + 1
    assert isinstance(ds.divisions[0], type(s.index[0]))
    tm.assert_series_equal(s, ds.compute())