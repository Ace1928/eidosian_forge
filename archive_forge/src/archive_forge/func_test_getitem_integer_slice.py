from __future__ import annotations
import contextlib
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import tokenize
from dask.dataframe._compat import PANDAS_GE_210, PANDAS_GE_220, IndexingError, tm
from dask.dataframe.indexing import _coerce_loc_index
from dask.dataframe.utils import assert_eq, make_meta, pyarrow_strings_enabled
def test_getitem_integer_slice():
    df = pd.DataFrame({'A': range(6)})
    ddf = dd.from_pandas(df, 2)
    with pytest.raises(NotImplementedError):
        ddf[1:3]
    df = pd.DataFrame({'A': range(6)}, index=[1.0, 2.0, 3.0, 5.0, 10.0, 11.0])
    ddf = dd.from_pandas(df, 2)
    ctx = contextlib.nullcontext()
    if PANDAS_GE_210:
        ctx = pytest.warns(FutureWarning, match='float-dtype index')
    with ctx:
        assert_eq(ddf[2:8], df[2:8])
    with ctx:
        assert_eq(ddf[2:], df[2:])
    with ctx:
        assert_eq(ddf[:8], df[:8])