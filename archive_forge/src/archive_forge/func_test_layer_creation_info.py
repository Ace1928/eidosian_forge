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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="doesn't make sense")
def test_layer_creation_info(tmpdir, engine):
    df = pd.DataFrame({'a': range(10), 'b': ['cat', 'dog'] * 5})
    dd.from_pandas(df, npartitions=1).to_parquet(tmpdir, engine=engine, partition_on=['b'])
    filters = [('b', '==', 'cat')]
    ddf1 = dd.read_parquet(tmpdir, engine=engine, filters=filters)
    assert 'dog' not in ddf1['b'].compute()
    ddf2 = dd.read_parquet(tmpdir, engine=engine)
    with pytest.raises(AssertionError):
        assert_eq(ddf1, ddf2)
    info = ddf2.dask.layers[ddf2._name].creation_info
    kwargs = info.get('kwargs', {})
    kwargs['filters'] = filters
    ddf3 = info['func'](*info.get('args', []), **kwargs)
    assert_eq(ddf1, ddf3)