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
def test_from_delayed_to_dask_array():
    from dask.blockwise import optimize_blockwise
    dfs = [delayed(pd.DataFrame)(np.ones((3, 2))) for i in range(3)]
    ddf = dd.from_delayed(dfs)
    arr = ddf.to_dask_array()
    keys = [k[0] for k in arr.__dask_keys__()]
    dsk = optimize_blockwise(arr.dask, keys=keys)
    dsk.cull(keys)
    result = arr.compute()
    assert result.shape == (9, 2)