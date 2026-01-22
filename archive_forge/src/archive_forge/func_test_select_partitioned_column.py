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
def test_select_partitioned_column(tmpdir, engine):
    fn = str(tmpdir)
    size = 20
    d = {'signal1': np.random.normal(0, 0.3, size=size).cumsum() + 50, 'fake_categorical1': np.random.choice(['A', 'B', 'C'], size=size), 'fake_categorical2': np.random.choice(['D', 'E', 'F'], size=size)}
    df = dd.from_pandas(pd.DataFrame(d), 2)
    df.to_parquet(fn, compression='snappy', write_index=False, engine=engine, partition_on=['fake_categorical1', 'fake_categorical2'])
    df_partitioned = dd.read_parquet(fn, engine=engine)
    df_partitioned[df_partitioned.fake_categorical1 == 'A'].compute()