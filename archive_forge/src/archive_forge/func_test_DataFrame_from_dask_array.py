from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import dask
import dask.array as da
import dask.dataframe as dd
from dask import config
from dask.blockwise import Blockwise
from dask.dataframe._compat import PANDAS_GE_200, tm
from dask.dataframe.io.io import _meta_from_array
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import assert_eq, get_string_dtype, pyarrow_strings_enabled
from dask.delayed import Delayed, delayed
from dask.utils_test import hlg_layer_topological
def test_DataFrame_from_dask_array():
    x = da.ones((10, 3), chunks=(4, 2))
    pdf = pd.DataFrame(np.ones((10, 3)), columns=['a', 'b', 'c'])
    df = dd.from_dask_array(x, ['a', 'b', 'c'])
    if not DASK_EXPR_ENABLED:
        assert not hlg_layer_topological(df.dask, -1).is_materialized()
    assert_eq(df, pdf)
    df2 = dd.from_array(x, columns=['a', 'b', 'c'])
    if not DASK_EXPR_ENABLED:
        assert not hlg_layer_topological(df2.dask, -1).is_materialized()
    assert_eq(df, df2)