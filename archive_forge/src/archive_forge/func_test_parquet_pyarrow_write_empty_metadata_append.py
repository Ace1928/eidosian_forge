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
def test_parquet_pyarrow_write_empty_metadata_append(tmpdir):
    tmpdir = str(tmpdir)
    df_a = dask.delayed(pd.DataFrame.from_dict)({'x': [1, 1, 2, 2], 'y': [1, 0, 1, 0]}, dtype=('int64', 'int64'))
    df_b = dask.delayed(pd.DataFrame.from_dict)({'x': [1, 2, 1, 2], 'y': [2, 0, 2, 0]}, dtype=('int64', 'int64'))
    df1 = dd.from_delayed([df_a, df_b])
    df1.to_parquet(tmpdir, partition_on=['x'], append=False, write_metadata_file=True)
    df_c = dask.delayed(pd.DataFrame.from_dict)({'x': [], 'y': []}, dtype=('int64', 'int64'))
    df_d = dask.delayed(pd.DataFrame.from_dict)({'x': [3, 3, 4, 4], 'y': [1, 0, 1, 0]}, dtype=('int64', 'int64'))
    df2 = dd.from_delayed([df_c, df_d])
    df2.to_parquet(tmpdir, partition_on=['x'], append=True, ignore_divisions=True, write_metadata_file=True)