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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="doesn't make sense")
@PYARROW_MARK
def test_retries_on_remote_filesystem(tmpdir):
    fn = str(tmpdir)
    remote_fn = f'simplecache://{tmpdir}'
    storage_options = {'target_protocol': 'file'}
    df = pd.DataFrame({'a': range(10)})
    ddf = dd.from_pandas(df, npartitions=2)
    ddf.to_parquet(fn)
    scalar = ddf.to_parquet(remote_fn, compute=False, storage_options=storage_options)
    layer = hlg_layer(scalar.dask, 'to-parquet')
    assert layer.annotations
    assert layer.annotations['retries'] == 5
    ddf2 = dd.read_parquet(remote_fn, storage_options=storage_options)
    layer = hlg_layer(ddf2.dask, 'read-parquet')
    assert layer.annotations
    assert layer.annotations['retries'] == 5
    scalar = ddf.to_parquet(fn, compute=False, storage_options=storage_options)
    layer = hlg_layer(scalar.dask, 'to-parquet')
    assert not layer.annotations
    ddf2 = dd.read_parquet(fn, storage_options=storage_options)
    layer = hlg_layer(ddf2.dask, 'read-parquet')
    assert not layer.annotations
    with dask.annotate(retries=2):
        scalar = ddf.to_parquet(remote_fn, compute=False, storage_options=storage_options)
        layer = hlg_layer(scalar.dask, 'to-parquet')
        assert layer.annotations
        assert layer.annotations['retries'] == 2
        ddf2 = dd.read_parquet(remote_fn, storage_options=storage_options)
        layer = hlg_layer(ddf2.dask, 'read-parquet')
        assert layer.annotations
        assert layer.annotations['retries'] == 2