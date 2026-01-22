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
@pytest.mark.skipif(not PANDAS_GE_200, reason='dataframe.convert-string requires pandas>=2.0')
def test_from_pandas_convert_string_config():
    pytest.importorskip('pyarrow', reason='Requires pyarrow strings')
    with dask.config.set({'dataframe.convert-string': False}):
        s = pd.Series(['foo', 'bar', 'ricky', 'bobby'], index=['a', 'b', 'c', 'd'])
        df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [5.0, 6.0, 7.0, 8.0], 'z': ['foo', 'bar', 'ricky', 'bobby']}, index=['a', 'b', 'c', 'd'])
        ds = dd.from_pandas(s, npartitions=2)
        ddf = dd.from_pandas(df, npartitions=2)
    assert_eq(s, ds)
    assert_eq(df, ddf)
    with dask.config.set({'dataframe.convert-string': True}):
        ds = dd.from_pandas(s, npartitions=2)
        ddf = dd.from_pandas(df, npartitions=2)
    s_pyarrow = s.astype('string[pyarrow]')
    s_pyarrow.index = s_pyarrow.index.astype('string[pyarrow]')
    df_pyarrow = df.astype({'z': 'string[pyarrow]'})
    df_pyarrow.index = df_pyarrow.index.astype('string[pyarrow]')
    assert_eq(s_pyarrow, ds)
    assert_eq(df_pyarrow, ddf)