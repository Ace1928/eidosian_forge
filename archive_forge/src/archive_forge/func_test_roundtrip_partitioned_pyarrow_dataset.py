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
def test_roundtrip_partitioned_pyarrow_dataset(tmpdir, engine):
    if engine == 'fastparquet' and PANDAS_GE_200:
        pytest.xfail('fastparquet reads as int64 while pyarrow does as int32')
    import pyarrow.parquet as pq
    from pyarrow.dataset import HivePartitioning, write_dataset
    df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    dask_path = tmpdir.mkdir('foo-dask')
    ddf = dd.from_pandas(df, npartitions=2)
    ddf.to_parquet(dask_path, engine=engine, partition_on=['col1'], write_index=False)
    pa_path = tmpdir.mkdir('foo-pyarrow')
    table = pa.Table.from_pandas(df)
    write_dataset(data=table, base_dir=pa_path, basename_template='part.{i}.parquet', format='parquet', partitioning=HivePartitioning(pa.schema([('col1', pa.int32())])))

    def _prep(x):
        return x.sort_values('col2')[['col1', 'col2']]
    df_read_dask = dd.read_parquet(dask_path, engine=engine)
    df_read_pa = pq.read_table(dask_path).to_pandas()
    assert_eq(_prep(df_read_dask), _prep(df_read_pa), check_index=False)
    df_read_dask = dd.read_parquet(pa_path, engine=engine)
    df_read_pa = pq.read_table(pa_path).to_pandas()
    assert_eq(_prep(df_read_dask), _prep(df_read_pa), check_index=False)