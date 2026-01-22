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
def test_not_in_predicate(tmp_path, engine):
    ddf = dd.from_dict({'A': range(8), 'B': [1, 1, 2, 2, 3, 3, 4, 4]}, npartitions=4)
    ddf.to_parquet(tmp_path, engine=engine)
    filters = [[('B', 'not in', (1, 2))]]
    result = dd.read_parquet(tmp_path, engine=engine, filters=filters)
    expected = pd.read_parquet(tmp_path, engine=engine, filters=filters)
    assert_eq(result, expected, check_index=False)
    with pytest.raises(ValueError, match='not a valid operator in predicates'):
        unsupported_op = [[('B', 'not eq', 1)]]
        dd.read_parquet(tmp_path, engine=engine, filters=unsupported_op).compute()