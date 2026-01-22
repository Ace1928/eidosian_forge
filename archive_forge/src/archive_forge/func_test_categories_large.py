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
@FASTPARQUET_MARK
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_categories_large(tmpdir, engine):
    fn = str(tmpdir.join('parquet_int16.parq'))
    numbers = np.random.randint(0, 800000, size=1000000)
    df = pd.DataFrame(numbers.T, columns=['name'])
    df.name = df.name.astype('category')
    df.to_parquet(fn, engine='fastparquet', compression='uncompressed')
    ddf = dd.read_parquet(fn, engine=engine, categories={'name': 80000})
    assert_eq(sorted(df.name.cat.categories), sorted(ddf.compute().name.cat.categories))