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
def test_pyarrow_directory_partitioning(tmpdir):
    path1 = tmpdir.mkdir('a')
    path2 = tmpdir.mkdir('b')
    path1 = os.path.join(path1, 'data.parquet')
    path2 = os.path.join(path2, 'data.parquet')
    _df1 = pd.DataFrame({'part': 'a', 'col': range(5)})
    _df2 = pd.DataFrame({'part': 'b', 'col': range(5)})
    _df1.to_parquet(path1)
    _df2.to_parquet(path2)
    expect = pd.concat([_df1, _df2], ignore_index=True)
    result = dd.read_parquet(str(tmpdir), dataset={'partitioning': ['part'], 'partition_base_dir': str(tmpdir)})
    result['part'] = result['part'].astype('object')
    assert_eq(result[list(expect.columns)], expect, check_index=False)