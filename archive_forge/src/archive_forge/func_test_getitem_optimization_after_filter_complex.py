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
def test_getitem_optimization_after_filter_complex(tmpdir, engine):
    df = pd.DataFrame({'a': [1, 2, 3] * 5, 'b': range(15), 'c': range(15)})
    dd.from_pandas(df, npartitions=3).to_parquet(tmpdir, engine=engine)
    ddf = dd.read_parquet(tmpdir, engine=engine)
    df2 = df[['b']]
    df2 = df2.assign(d=1)
    df2 = df[df2['d'] == 1][['b']]
    ddf2 = ddf[['b']]
    ddf2 = ddf2.assign(d=1)
    ddf2 = ddf[ddf2['d'] == 1][['b']]
    dsk = optimize_dataframe_getitem(ddf2.dask, keys=[ddf2._name])
    subgraph_rd = hlg_layer(dsk, 'read-parquet')
    assert isinstance(subgraph_rd, DataFrameIOLayer)
    assert set(subgraph_rd.columns) == {'b'}
    assert_eq(df2, ddf2)