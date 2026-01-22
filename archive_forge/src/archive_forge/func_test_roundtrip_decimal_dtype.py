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
def test_roundtrip_decimal_dtype(tmpdir):
    tmpdir = str(tmpdir)
    data = [{'ts': pd.to_datetime('2021-01-01', utc='Europe/Berlin'), 'col1': Decimal('123.00')} for i in range(23)]
    ddf1 = dd.from_pandas(pd.DataFrame(data), npartitions=1)
    ddf1.to_parquet(path=tmpdir, schema={'col1': pa.decimal128(5, 2)})
    ddf2 = dd.read_parquet(tmpdir)
    if pyarrow_strings_enabled():
        assert pa.types.is_decimal(ddf2['col1'].dtype.pyarrow_dtype)
        ddf1 = ddf1.astype({'col1': pd.ArrowDtype(pa.decimal128(5, 2))})
    else:
        assert ddf1['col1'].dtype == ddf2['col1'].dtype
    assert_eq(ddf1, ddf2, check_divisions=False)