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
@pytest.mark.parametrize('filter_value', ({1}, [1], (1,)), ids=('set', 'list', 'tuple'))
def test_in_predicate_can_use_iterables(tmp_path, engine, filter_value):
    """Regression test for https://github.com/dask/dask/issues/8720"""
    path = tmp_path / 'in_predicate_iterable_pandas.parquet'
    df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [1, 1, 2, 2]})
    df.to_parquet(path, engine=engine)
    filters = [('B', 'in', filter_value)]
    result = dd.read_parquet(path, engine=engine, filters=filters)
    expected = pd.read_parquet(path, engine=engine, filters=filters)
    assert_eq(result, expected)
    ddf = dd.from_pandas(df, npartitions=2)
    path = tmp_path / 'in_predicate_iterable_dask.parquet'
    ddf.to_parquet(path, engine=engine)
    result = dd.read_parquet(path, engine=engine, filters=filters)
    expected = pd.read_parquet(path, engine=engine, filters=filters)
    assert_eq(result, expected, check_index=False)