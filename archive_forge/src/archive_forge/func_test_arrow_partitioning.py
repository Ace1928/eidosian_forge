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
def test_arrow_partitioning(tmpdir):
    path = str(tmpdir)
    data = {'p': np.repeat(np.arange(3), 2).astype(np.int8), 'b': np.repeat(-1, 6).astype(np.int16), 'c': np.repeat(-2, 6).astype(np.float32), 'd': np.repeat(-3, 6).astype(np.float64)}
    pdf = pd.DataFrame(data)
    ddf = dd.from_pandas(pdf, npartitions=2)
    ddf.to_parquet(path, write_index=False, partition_on='p')
    ddf = dd.read_parquet(path, index=False)
    ddf.astype({'b': np.float32}).compute()