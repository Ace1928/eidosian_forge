from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.parametrize('backend', ['pandas', pytest.param('cudf', marks=pytest.mark.gpu)])
def test_groupby_large_ints_exception(backend):
    data_source = pytest.importorskip(backend)
    if backend == 'cudf':
        dask_cudf = pytest.importorskip('dask_cudf')
        data_frame = dask_cudf.from_cudf
    else:
        data_frame = dd.from_pandas
    max = np.iinfo(np.uint64).max
    sqrt = max ** 0.5
    series = data_source.Series(np.concatenate([sqrt * np.arange(5), np.arange(35)])).astype('int64')
    df = data_source.DataFrame({'x': series, 'z': np.arange(40), 'y': np.arange(40)})
    ddf = data_frame(df, npartitions=1)
    assert_eq(df.groupby('x').std(), ddf.groupby('x').std().compute(scheduler='single-threaded'))