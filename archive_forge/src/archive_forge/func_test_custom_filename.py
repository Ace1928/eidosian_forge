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
def test_custom_filename(tmpdir, engine):
    fn = str(tmpdir)
    pdf = pd.DataFrame({'num1': [1, 2, 3, 4], 'num2': [7, 8, 9, 10]})
    df = dd.from_pandas(pdf, npartitions=2)
    df.to_parquet(fn, write_metadata_file=True, name_function=lambda x: f'hi-{x}.parquet', engine=engine)
    files = os.listdir(fn)
    assert '_common_metadata' in files
    assert '_metadata' in files
    assert 'hi-0.parquet' in files
    assert 'hi-1.parquet' in files
    assert_eq(df, dd.read_parquet(fn, engine=engine, calculate_divisions=True))