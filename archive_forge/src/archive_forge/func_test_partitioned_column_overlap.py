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
@pytest.mark.parametrize('write_cols', [['part', 'col'], ['part', 'kind', 'col']])
def test_partitioned_column_overlap(tmpdir, engine, write_cols):
    tmpdir.mkdir('part=a')
    tmpdir.mkdir('part=b')
    path0 = str(tmpdir.mkdir('part=a/kind=x'))
    path1 = str(tmpdir.mkdir('part=b/kind=x'))
    path0 = os.path.join(path0, 'data.parquet')
    path1 = os.path.join(path1, 'data.parquet')
    _df1 = pd.DataFrame({'part': 'a', 'kind': 'x', 'col': range(5)})
    _df2 = pd.DataFrame({'part': 'b', 'kind': 'x', 'col': range(5)})
    df1 = _df1[write_cols]
    df2 = _df2[write_cols]
    df1.to_parquet(path0, index=False)
    df2.to_parquet(path1, index=False)
    if engine == 'fastparquet':
        path = [path0, path1]
    else:
        path = str(tmpdir)
    expect = pd.concat([_df1, _df2], ignore_index=True)
    if engine == 'fastparquet' and fastparquet_version > parse_version('0.8.3'):
        result = dd.read_parquet(path, engine=engine)
        assert result.compute().reset_index(drop=True).to_dict() == expect.to_dict()
    elif write_cols == ['part', 'kind', 'col']:
        result = dd.read_parquet(path, engine=engine)
        assert_eq(result, expect, check_index=False)
    else:
        with pytest.raises(ValueError):
            dd.read_parquet(path, engine=engine)