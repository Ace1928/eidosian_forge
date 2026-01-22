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
def test_partitioned_preserve_index(tmpdir, write_engine, read_engine):
    tmp = str(tmpdir)
    size = 1000
    npartitions = 4
    b = np.arange(npartitions, dtype='int32').repeat(size // npartitions)
    data = pd.DataFrame({'myindex': np.arange(size), 'A': np.random.random(size=size), 'B': pd.Categorical(b)}).set_index('myindex')
    data.index.name = None
    df1 = dd.from_pandas(data, npartitions=npartitions)
    df1.to_parquet(tmp, partition_on='B', engine=write_engine)
    expect = data[data['B'] == 1]
    if PANDAS_GE_200 and read_engine == 'fastparquet':
        expect = expect.copy()
        expect['B'] = expect['B'].astype(pd.CategoricalDtype(expect['B'].dtype.categories.astype('int64')))
    got = dd.read_parquet(tmp, engine=read_engine, filters=[('B', '==', 1)])
    assert_eq(expect, got)