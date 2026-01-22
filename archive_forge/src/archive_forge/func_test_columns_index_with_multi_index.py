from __future__ import annotations
import contextlib
import glob
import math
import os
import sys
import warnings
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask
import dask.dataframe as dd
import dask.multiprocessing
from dask.array.numpy_compat import NUMPY_GE_124
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import (
from dask.dataframe.io.parquet.core import get_engine
from dask.dataframe.io.parquet.utils import _parse_pandas_metadata
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
from dask.layers import DataFrameIOLayer
from dask.utils import natural_sort_key
from dask.utils_test import hlg_layer
@pytest.mark.xfail(PANDAS_GE_300, reason='KeyError')
def test_columns_index_with_multi_index(tmpdir, engine):
    fn = os.path.join(str(tmpdir), 'test.parquet')
    index = pd.MultiIndex.from_arrays([np.arange(10), np.arange(10) + 1], names=['x0', 'x1'])
    df = pd.DataFrame(np.random.randn(10, 2), columns=['a', 'b'], index=index)
    df2 = df.reset_index(drop=False)
    if engine == 'fastparquet':
        fastparquet.write(fn, df.reset_index(), write_index=False)
    else:
        pq.write_table(pa.Table.from_pandas(df.reset_index(), preserve_index=False), fn)
    ddf = dd.read_parquet(fn, engine=engine, index=index.names)
    assert_eq(ddf, df)
    d = dd.read_parquet(fn, columns='a', engine=engine, index=index.names)
    assert_eq(d, df['a'])
    d = dd.read_parquet(fn, index=['a', 'b'], columns=['x0', 'x1'], engine=engine)
    assert_eq(d, df2.set_index(['a', 'b'])[['x0', 'x1']])
    d = dd.read_parquet(fn, index=False, engine=engine)
    assert_eq(d, df2)
    d = dd.read_parquet(fn, columns=['b'], index=['a'], engine=engine)
    assert_eq(d, df2.set_index('a')[['b']])
    d = dd.read_parquet(fn, columns=['a', 'b'], index=['x0'], engine=engine)
    assert_eq(d, df2.set_index('x0')[['a', 'b']])
    d = dd.read_parquet(fn, columns=['x0', 'a'], index=['x1'], engine=engine)
    assert_eq(d, df2.set_index('x1')[['x0', 'a']])
    d = dd.read_parquet(fn, index=False, columns=['x0', 'b'], engine=engine)
    assert_eq(d, df2[['x0', 'b']])
    for index in ['x1', 'b']:
        d = dd.read_parquet(fn, index=index, columns=['x0', 'a'], engine=engine)
        assert_eq(d, df2.set_index(index)[['x0', 'a']])
    for index in ['a', 'x0']:
        with pytest.raises((ValueError, KeyError)):
            d = dd.read_parquet(fn, index=index, columns=['x0', 'a'], engine=engine)
    for ind, col, sol_df in [('x1', 'x0', df2.set_index('x1')), (False, 'b', df2), (False, 'x0', df2[['x0']]), ('a', 'x0', df2.set_index('a')[['x0']]), ('a', 'b', df2.set_index('a'))]:
        d = dd.read_parquet(fn, index=ind, columns=col, engine=engine)
        assert_eq(d, sol_df[col])