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
def test_partition_on(tmpdir, engine):
    tmpdir = str(tmpdir)
    df = pd.DataFrame({'a1': np.random.choice(['A', 'B', 'C'], size=100), 'a2': np.random.choice(['X', 'Y', 'Z'], size=100), 'b': np.random.random(size=100), 'c': np.random.randint(1, 5, size=100), 'd': np.arange(0, 100)})
    d = dd.from_pandas(df, npartitions=2)
    d.to_parquet(tmpdir, partition_on=['a1', 'a2'], engine=engine)
    out = dd.read_parquet(tmpdir, engine=engine, index=False, calculate_divisions=False).compute()
    for val in df.a1.unique():
        assert set(df.d[df.a1 == val]) == set(out.d[out.a1 == val])
    out = dd.read_parquet(tmpdir, engine=engine, columns=['d', 'a2']).compute()
    for val in df.a2.unique():
        assert set(df.d[df.a2 == val]) == set(out.d[out.a2 == val])