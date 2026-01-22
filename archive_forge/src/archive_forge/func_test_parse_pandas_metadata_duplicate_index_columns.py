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
def test_parse_pandas_metadata_duplicate_index_columns():
    md = {'column_indexes': [{'field_name': None, 'metadata': {'encoding': 'UTF-8'}, 'name': None, 'numpy_type': 'object', 'pandas_type': 'unicode'}], 'columns': [{'field_name': 'A', 'metadata': None, 'name': 'A', 'numpy_type': 'int64', 'pandas_type': 'int64'}, {'field_name': '__index_level_0__', 'metadata': None, 'name': 'A', 'numpy_type': 'object', 'pandas_type': 'unicode'}], 'index_columns': ['__index_level_0__'], 'pandas_version': '0.21.0'}
    index_names, column_names, storage_name_mapping, column_index_names = _parse_pandas_metadata(md)
    assert index_names == ['A']
    assert column_names == ['A']
    assert storage_name_mapping == {'__index_level_0__': 'A', 'A': 'A'}
    assert column_index_names == [None]