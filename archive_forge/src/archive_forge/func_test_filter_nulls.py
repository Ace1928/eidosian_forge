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
@pytest.mark.parametrize('filters,op,length', [pytest.param([('c', '!=', 'a')], lambda x: x[x['c'] != 'a'], 13, marks=pytest.mark.xfail_with_pyarrow_strings), ([('c', '==', 'a')], lambda x: x[x['c'] == 'a'], 2)])
@pytest.mark.parametrize('split_row_groups', [True, False])
def test_filter_nulls(tmpdir, filters, op, length, split_row_groups, engine):
    if engine == 'pyarrow' and parse_version(pa.__version__) < parse_version('8.0.0'):
        pytest.skip('pyarrow>=8.0.0 needed for correct null filtering')
    path = tmpdir.join('test.parquet')
    df = pd.DataFrame({'a': [1, None] * 5 + [None] * 5, 'b': np.arange(14).tolist() + [None], 'c': ['a', None] * 2 + [None] * 11})
    df.to_parquet(path, engine='pyarrow', row_group_size=10)
    result = dd.read_parquet(path, engine=engine, filters=filters, split_row_groups=split_row_groups)
    assert len(op(result)) == length
    assert_eq(op(result), op(df), check_index=False)