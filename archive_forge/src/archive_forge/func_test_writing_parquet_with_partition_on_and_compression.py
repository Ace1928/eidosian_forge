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
@pytest.mark.parametrize('compression,', [None, 'gzip', 'snappy'])
def test_writing_parquet_with_partition_on_and_compression(tmpdir, compression, engine):
    fn = str(tmpdir)
    df = pd.DataFrame({'x': ['a', 'b', 'c'] * 10, 'y': [1, 2, 3] * 10})
    df.index.name = 'index'
    ddf = dd.from_pandas(df, npartitions=3)
    ddf.to_parquet(fn, compression=compression, engine=engine, partition_on=['x'], write_metadata_file=True)
    check_compression(engine, fn, compression)