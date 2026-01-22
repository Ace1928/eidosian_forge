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
def test_to_bag():
    a = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [2, 3, 4, 5]}, index=pd.Index([1.0, 2.0, 3.0, 4.0], name='ind'))
    ddf = dd.from_pandas(a, 2)
    assert ddf.to_bag().compute() == list(a.itertuples(False))
    assert ddf.to_bag(True).compute() == list(a.itertuples(True))
    assert ddf.to_bag(format='dict').compute() == [{'x': 'a', 'y': 2}, {'x': 'b', 'y': 3}, {'x': 'c', 'y': 4}, {'x': 'd', 'y': 5}]
    assert ddf.to_bag(True, format='dict').compute() == [{'index': 1.0, 'x': 'a', 'y': 2}, {'index': 2.0, 'x': 'b', 'y': 3}, {'index': 3.0, 'x': 'c', 'y': 4}, {'index': 4.0, 'x': 'd', 'y': 5}]
    assert ddf.x.to_bag(True).compute() == list(a.x.items())
    assert ddf.x.to_bag().compute() == list(a.x)
    assert ddf.x.to_bag(True, format='dict').compute() == [{'x': 'a'}, {'x': 'b'}, {'x': 'c'}, {'x': 'd'}]
    assert ddf.x.to_bag(format='dict').compute() == [{'x': 'a'}, {'x': 'b'}, {'x': 'c'}, {'x': 'd'}]