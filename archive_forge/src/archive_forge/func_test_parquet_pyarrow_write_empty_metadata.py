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
def test_parquet_pyarrow_write_empty_metadata(tmpdir):
    tmpdir = str(tmpdir)
    df_a = dask.delayed(pd.DataFrame.from_dict)({'x': [], 'y': []}, dtype=('int', 'int'))
    df_b = dask.delayed(pd.DataFrame.from_dict)({'x': [1, 1, 2, 2], 'y': [1, 0, 1, 0]}, dtype=('int64', 'int64'))
    df_c = dask.delayed(pd.DataFrame.from_dict)({'x': [1, 2, 1, 2], 'y': [1, 0, 1, 0]}, dtype=('int64', 'int64'))
    df = dd.from_delayed([df_a, df_b, df_c])
    df.to_parquet(tmpdir, partition_on=['x'], append=False, write_metadata_file=True)
    files = os.listdir(tmpdir)
    assert '_metadata' in files
    assert '_common_metadata' in files
    schema_common = pq.ParquetFile(os.path.join(tmpdir, '_common_metadata')).schema.to_arrow_schema()
    pandas_metadata = schema_common.pandas_metadata
    assert pandas_metadata
    assert pandas_metadata.get('index_columns', False)