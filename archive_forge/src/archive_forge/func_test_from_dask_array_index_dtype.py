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
def test_from_dask_array_index_dtype():
    x = da.ones((10,), chunks=(5,))
    df = pd.DataFrame({'date': pd.date_range('2019-01-01', periods=10, freq='1min'), 'val1': list(range(10))})
    ddf = dd.from_pandas(df, npartitions=2).set_index('date')
    ddf2 = dd.from_dask_array(x, index=ddf.index, columns='val2')
    assert ddf.index.dtype == ddf2.index.dtype
    assert ddf.index.name == ddf2.index.name
    df = pd.DataFrame({'idx': np.arange(0, 1, 0.1), 'val1': list(range(10))})
    ddf = dd.from_pandas(df, npartitions=2).set_index('idx')
    ddf2 = dd.from_dask_array(x, index=ddf.index, columns='val2')
    assert ddf.index.dtype == ddf2.index.dtype
    assert ddf.index.name == ddf2.index.name