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
@pytest.mark.parametrize('chunksizes, expected_divisions', [pytest.param((1, 2, 3, 0), (0, 1, 3, 5, 5)), pytest.param((0, 1, 2, 3), (0, 0, 1, 3, 5)), pytest.param((1, 0, 2, 3), (0, 1, 1, 3, 5))])
def test_from_dask_array_empty_chunks(chunksizes, expected_divisions):
    monotonic_index = da.from_array(np.arange(6), chunks=chunksizes)
    df = dd.from_dask_array(monotonic_index)
    assert df.divisions == expected_divisions