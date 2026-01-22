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
def test_pandas_metadata_nullable_pyarrow(tmpdir):
    tmpdir = str(tmpdir)
    ddf1 = dd.from_pandas(pd.DataFrame({'A': pd.array([1, None, 2], dtype='Int64'), 'B': pd.array(['dog', 'cat', None], dtype='str')}), npartitions=1)
    ddf1.to_parquet(tmpdir)
    ddf2 = dd.read_parquet(tmpdir, calculate_divisions=True)
    assert_eq(ddf1, ddf2, check_index=False)