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
def test_custom_filename_with_partition(tmpdir, engine):
    fn = str(tmpdir)
    pdf = pd.DataFrame({'first_name': ['frank', 'li', 'marcela', 'luis'], 'country': ['canada', 'china', 'venezuela', 'venezuela']})
    df = dd.from_pandas(pdf, npartitions=4)
    df.to_parquet(fn, engine=engine, partition_on=['country'], name_function=lambda x: f'{x}-cool.parquet', write_index=False)
    for _, dirs, files in os.walk(fn):
        for dir in dirs:
            assert dir in ('country=canada', 'country=china', 'country=venezuela')
        for file in files:
            assert file in (*[f'{i}-cool.parquet' for i in range(df.npartitions)], '_common_metadata', '_metadata')
    actual = dd.read_parquet(fn, engine=engine, index=False)
    assert_eq(pdf, actual, check_index=False, check_dtype=False, check_categorical=False)