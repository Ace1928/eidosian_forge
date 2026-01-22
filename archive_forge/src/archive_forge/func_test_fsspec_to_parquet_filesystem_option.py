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
def test_fsspec_to_parquet_filesystem_option(tmp_path):
    from fsspec import get_filesystem_class
    key1 = '/read1'
    key2 = str(tmp_path / 'write1')
    df = pd.DataFrame({'a': range(10)})
    fs = get_filesystem_class('memory')(use_instance_cache=False)
    df.to_parquet(key1, filesystem=fs)
    ddf = dd.read_parquet(key1, filesystem=fs)
    assert_eq(ddf, df)
    ddf.to_parquet(key2, filesystem=fs)
    assert len(list(tmp_path.iterdir())) == 0, 'wrote to local fs'
    assert len(fs.ls(key2, detail=False)) == 1
    ddf.to_parquet(key2, append=True, filesystem=fs)
    assert len(fs.ls(key2, detail=False)) == 2, 'should have two parts'
    rddf = dd.read_parquet(key2, filesystem=fs)
    assert_eq(rddf, dd.concat([ddf, ddf]))