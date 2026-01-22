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
def test_filters_v0(tmpdir, write_engine, read_engine):
    fn = str(tmpdir)
    df = pd.DataFrame({'at': ['ab', 'aa', 'ba', 'da', 'bb']})
    ddf = dd.from_pandas(df, npartitions=1)
    ddf.repartition(npartitions=1, force=True).to_parquet(fn, write_index=False, engine=write_engine)
    ddf2 = dd.read_parquet(fn, index=False, engine=read_engine, filters=[('at', '==', 'aa')]).compute()
    ddf3 = dd.read_parquet(fn, index=False, engine=read_engine, filters=[('at', '=', 'aa')]).compute()
    if read_engine == 'pyarrow':
        assert_eq(ddf2, ddf[ddf['at'] == 'aa'], check_index=False)
        assert_eq(ddf3, ddf[ddf['at'] == 'aa'], check_index=False)
    else:
        assert_eq(ddf2, ddf)
        assert_eq(ddf3, ddf)
    ddf.repartition(npartitions=2, force=True).to_parquet(fn, engine=write_engine)
    ddf2 = dd.read_parquet(fn, engine=read_engine).compute()
    assert_eq(ddf2, ddf)
    if read_engine == 'fastparquet':
        ddf.repartition(npartitions=2, force=True).to_parquet(fn, engine=write_engine)
        df2 = fastparquet.ParquetFile(fn).to_pandas(filters=[('at', '==', 'aa')])
        df3 = fastparquet.ParquetFile(fn).to_pandas(filters=[('at', '=', 'aa')])
        assert len(df2) > 0
        assert len(df3) > 0
    ddf.repartition(npartitions=2, force=True).to_parquet(fn, engine=write_engine)
    ddf2 = dd.read_parquet(fn, engine=read_engine, filters=[('at', '==', 'aa')]).compute()
    ddf3 = dd.read_parquet(fn, engine=read_engine, filters=[('at', '=', 'aa')]).compute()
    assert len(ddf2) > 0
    assert len(ddf3) > 0
    assert_eq(ddf2, ddf3)