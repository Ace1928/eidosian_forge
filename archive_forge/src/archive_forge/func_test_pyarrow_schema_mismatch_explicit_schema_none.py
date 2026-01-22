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
def test_pyarrow_schema_mismatch_explicit_schema_none(tmpdir):
    df1 = pd.DataFrame({'x': [1, 2, 3], 'y': [4.5, 6, 7]})
    df2 = pd.DataFrame({'x': [4, 5, 6], 'y': ['a', 'b', 'c']})
    ddf = dd.from_delayed([dask.delayed(df1), dask.delayed(df2)], meta=df1, verify_meta=False)
    ddf.to_parquet(str(tmpdir), schema=None)
    res = dd.read_parquet(tmpdir)
    sol = pd.concat([df1, df2])
    assert_eq(res, sol, check_dtype=False)