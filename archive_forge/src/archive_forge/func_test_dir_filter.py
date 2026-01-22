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
def test_dir_filter(tmpdir, engine):
    df = pd.DataFrame.from_dict({'A': {0: 351.0, 1: 355.0, 2: 358.0, 3: 266.0, 4: 266.0, 5: 268.0, 6: np.nan}, 'B': {0: 2063.0, 1: 2051.0, 2: 1749.0, 3: 4281.0, 4: 3526.0, 5: 3462.0, 6: np.nan}, 'year': {0: 2019, 1: 2019, 2: 2020, 3: 2020, 4: 2020, 5: 2020, 6: 2020}})
    ddf = dask.dataframe.from_pandas(df, npartitions=1)
    ddf.to_parquet(tmpdir, partition_on='year', engine=engine)
    ddf2 = dd.read_parquet(tmpdir, filters=[('year', '==', 2020)], engine=engine)
    ddf2['year'] = ddf2.year.astype('int64')
    assert_eq(ddf2, df[df.year == 2020])