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
@pytest.mark.skip_with_pyarrow_strings
def test_append_dict_column(tmpdir):
    """See: https://github.com/dask/dask/issues/7492
    Note: fastparquet engine is missing dict-column support
    """
    tmp = str(tmpdir)
    dts = pd.date_range('2020-01-01', '2021-01-01')
    df = pd.DataFrame({'value': [{'x': x} for x in range(len(dts))]}, index=dts)
    ddf1 = dd.from_pandas(df, npartitions=1)
    schema = {'value': pa.struct([('x', pa.int32())])}
    ddf1.to_parquet(tmp, append=True, schema=schema)
    ddf1.to_parquet(tmp, append=True, schema=schema, ignore_divisions=True)
    ddf2 = dd.read_parquet(tmp)
    expect = pd.concat([df, df])
    result = ddf2.compute()
    assert_eq(expect, result)