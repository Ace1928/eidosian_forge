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
@pytest.mark.parametrize('null_value', [None, pd.NaT, pd.NA])
def test_from_pandas_with_index_nulls(null_value):
    df = pd.DataFrame({'x': [1, 2, 3]}, index=['C', null_value, 'A'])
    with pytest.raises(NotImplementedError, match='is non-numeric and contains nulls'):
        dd.from_pandas(df, npartitions=2, sort=False)