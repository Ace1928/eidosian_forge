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
def test_pathlib_path(tmpdir, engine):
    import pathlib
    df = pd.DataFrame({'x': [4, 5, 6, 1, 2, 3]})
    df.index.name = 'index'
    ddf = dd.from_pandas(df, npartitions=2)
    path = pathlib.Path(str(tmpdir))
    ddf.to_parquet(path, engine=engine)
    ddf2 = dd.read_parquet(path, engine=engine, calculate_divisions=True)
    assert_eq(ddf, ddf2)