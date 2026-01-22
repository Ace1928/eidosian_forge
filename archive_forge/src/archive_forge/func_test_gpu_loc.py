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
@pytest.mark.gpu
def test_gpu_loc():
    cudf = pytest.importorskip('cudf')
    cupy = pytest.importorskip('cupy')
    index = [1, 5, 10, 11, 12, 100, 200, 300]
    df = cudf.DataFrame({'a': range(8), 'index': index}).set_index('index')
    ddf = dd.from_pandas(df, npartitions=3)
    cdf_index = cudf.Series([1, 100, 300])
    cupy_index = cupy.array([1, 100, 300])
    assert_eq(ddf.loc[cdf_index], df.loc[cupy_index])
    assert_eq(ddf.loc[cupy_index], df.loc[cupy_index])