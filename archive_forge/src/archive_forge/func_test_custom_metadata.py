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
def test_custom_metadata(tmpdir, engine):
    custom_metadata = {b'my_key': b'my_data'}
    path = str(tmpdir)
    df = pd.DataFrame({'a': range(10), 'b': range(10)})
    dd.from_pandas(df, npartitions=2).to_parquet(path, engine=engine, custom_metadata=custom_metadata, write_metadata_file=True)
    assert_eq(df, dd.read_parquet(path, engine=engine))
    if pq:
        files = glob.glob(os.path.join(path, '*.parquet'))
        files += [os.path.join(path, '_metadata')]
        for fn in files:
            _md = pq.ParquetFile(fn).metadata.metadata
            for k in custom_metadata.keys():
                assert _md[k] == custom_metadata[k]
    custom_metadata = {b'pandas': b'my_new_pandas_md'}
    with pytest.raises(ValueError) as e:
        dd.from_pandas(df, npartitions=2).to_parquet(path, engine=engine, custom_metadata=custom_metadata)
    assert 'User-defined key/value' in str(e.value)