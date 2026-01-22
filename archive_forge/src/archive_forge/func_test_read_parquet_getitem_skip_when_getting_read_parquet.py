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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='makes no sense')
def test_read_parquet_getitem_skip_when_getting_read_parquet(tmpdir, engine):
    pdf = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': ['a', 'b', 'c', 'd', 'e', 'f']})
    path = os.path.join(str(tmpdir), 'data.parquet')
    pdf.to_parquet(path, engine=engine)
    ddf = dd.read_parquet(path, engine=engine)
    a, b = dask.optimize(ddf['A'], ddf)
    ddf = ddf['A']
    dsk = optimize_dataframe_getitem(ddf.dask, keys=[(ddf._name, 0)])
    read = [key for key in dsk.layers if key.startswith('read-parquet')][0]
    subgraph = dsk.layers[read]
    assert isinstance(subgraph, DataFrameIOLayer)
    expected = ['A']
    assert subgraph.columns == expected