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
def test_from_dask_array_index_raises():
    x = da.random.uniform(size=(10,), chunks=(5,))
    with pytest.raises(ValueError, match='must be an instance'):
        dd.from_dask_array(x, index=pd.Index(np.arange(10)))
    a = dd.from_pandas(pd.Series(range(12)), npartitions=2)
    b = dd.from_pandas(pd.Series(range(12)), npartitions=4)
    with pytest.raises(ValueError, match='.*index.*numbers of blocks.*4 != 2'):
        dd.from_dask_array(a.values, index=b.index)