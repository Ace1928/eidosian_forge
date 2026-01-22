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
def test_read_write_overwrite_is_true(tmpdir, engine):
    ddf = dd.from_pandas(pd.DataFrame(np.random.randint(low=0, high=100, size=(100, 10)), columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']), npartitions=5)
    ddf = ddf.reset_index(drop=True)
    dd.to_parquet(ddf, tmpdir, engine=engine, overwrite=True)
    ddf2 = ddf.repartition(npartitions=3)
    dd.to_parquet(ddf2, tmpdir, engine=engine, overwrite=True)
    files = os.listdir(tmpdir)
    files = [f for f in files if f not in ['_common_metadata', '_metadata']]
    assert len(files) == ddf2.npartitions