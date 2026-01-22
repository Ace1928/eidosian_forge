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
@pytest.mark.parametrize('split_row_groups', [8, 25])
def test_split_row_groups_int_aggregate_files(tmpdir, engine, split_row_groups):
    row_group_size = 10
    size = 800
    df = pd.DataFrame({'i32': np.arange(size, dtype=np.int32), 'f': np.arange(size, dtype=np.float64)})
    dd.from_pandas(df, npartitions=4).to_parquet(str(tmpdir), engine='pyarrow', row_group_size=row_group_size, write_index=False)
    ddf2 = dd.read_parquet(str(tmpdir), engine=engine, split_row_groups=split_row_groups, aggregate_files=True)
    npartitions_expected = math.ceil(size / row_group_size / split_row_groups)
    assert ddf2.npartitions == npartitions_expected
    assert len(ddf2) == size
    assert_eq(df, ddf2, check_index=False)