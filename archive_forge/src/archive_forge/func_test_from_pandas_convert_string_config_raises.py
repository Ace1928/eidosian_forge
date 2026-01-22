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
@pytest.mark.skipif(PANDAS_GE_200, reason='Requires pandas<2.0')
def test_from_pandas_convert_string_config_raises():
    pytest.importorskip('pyarrow', reason='Different error without pyarrow')
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [5.0, 6.0, 7.0, 8.0], 'z': ['foo', 'bar', 'ricky', 'bobby']}, index=['a', 'b', 'c', 'd'])
    with dask.config.set({'dataframe.convert-string': True}):
        with pytest.raises(RuntimeError, match='requires `pandas>=2.0` to be installed'):
            dd.from_pandas(df, npartitions=2)