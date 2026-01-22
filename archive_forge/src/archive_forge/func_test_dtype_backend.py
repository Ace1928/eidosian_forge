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
@PYARROW_MARK
@pytest.mark.parametrize('dtype_backend', ['numpy_nullable', 'pyarrow'])
@pytest.mark.skipif(not PANDAS_GE_150, reason='Requires pyarrow-backed nullable dtypes')
def test_dtype_backend(tmp_path, dtype_backend, engine):
    """
    Test reading a parquet file without pandas metadata,
    but forcing use of nullable dtypes where appropriate
    """
    dtype_extra = '' if dtype_backend == 'numpy_nullable' else '[pyarrow]'
    df = pd.DataFrame({'a': pd.Series([1, 2, pd.NA, 3, 4], dtype=f'Int64{dtype_extra}'), 'b': pd.Series([True, pd.NA, False, True, False], dtype=f'boolean{dtype_extra}'), 'c': pd.Series([0.1, 0.2, 0.3, pd.NA, 0.4], dtype=f'Float64{dtype_extra}'), 'd': pd.Series(['a', 'b', 'c', 'd', pd.NA], dtype=f'string{dtype_extra}')})
    ddf = dd.from_pandas(df, npartitions=2)

    @dask.delayed
    def write_partition(df, i):
        """Write a parquet file without the pandas metadata"""
        table = pa.Table.from_pandas(df).replace_schema_metadata({})
        pq.write_table(table, tmp_path / f'part.{i}.parquet')
    partitions = ddf.to_delayed()
    dask.compute([write_partition(p, i) for i, p in enumerate(partitions)])
    if engine == 'fastparquet':
        with pytest.raises(ValueError, match='`dtype_backend` is not supported'):
            dd.read_parquet(tmp_path, engine=engine, dtype_backend=dtype_backend)
    else:
        ddf2 = dd.read_parquet(tmp_path, engine=engine, dtype_backend=dtype_backend)
        assert_eq(df, ddf2, check_index=False)