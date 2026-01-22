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
def test_empty_partition(tmpdir, engine):
    fn = str(tmpdir)
    df = pd.DataFrame({'a': range(10), 'b': range(10)})
    ddf = dd.from_pandas(df, npartitions=5)
    ddf2 = ddf[ddf.a <= 5]
    ddf2.to_parquet(fn, engine=engine)
    ddf3 = dd.read_parquet(fn, engine=engine, calculate_divisions=True)
    assert ddf3.npartitions < 5
    sol = ddf2.compute()
    assert_eq(sol, ddf3, check_names=False, check_index=False)