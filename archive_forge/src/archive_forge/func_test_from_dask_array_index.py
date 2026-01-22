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
@pytest.mark.parametrize('as_frame', [True, False])
def test_from_dask_array_index(as_frame):
    s = dd.from_pandas(pd.Series(range(10), index=list('abcdefghij')), npartitions=3)
    if as_frame:
        s = s.to_frame()
    result = dd.from_dask_array(s.values, index=s.index)
    assert_eq(s, result)