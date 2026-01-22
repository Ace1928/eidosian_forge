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
@pytest.mark.skipif(not PANDAS_GE_200, reason='Requires pd.ArrowDtype')
def test_dtype_backend_categoricals(tmp_path):
    df = pd.DataFrame({'a': pd.Series(['x', 'y'], dtype='category'), 'b': [1, 2]})
    outdir = tmp_path / 'out.parquet'
    df.to_parquet(outdir)
    ddf = dd.read_parquet(outdir, dtype_backend='pyarrow')
    pdf = pd.read_parquet(outdir, dtype_backend='pyarrow')
    assert_eq(ddf, pdf, sort_results=PANDAS_GE_202)