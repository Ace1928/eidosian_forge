from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('npartitions', [1, 4])
@pytest.mark.parametrize('use_dask_input', [True, False])
def test_map_overlap(npartitions, use_dask_input):
    ddf = df
    if use_dask_input:
        ddf = dd.from_pandas(df, npartitions)
    for before, after in [(0, 3), (3, 0), (3, 3), (0, 0)]:
        res = dd.map_overlap(shifted_sum, ddf, before, after, before, after, c=2)
        sol = shifted_sum(df, before, after, c=2)
        assert_eq(res, sol)
        res = dd.map_overlap(shifted_sum, ddf.b, before, after, before, after, c=2)
        sol = shifted_sum(df.b, before, after, c=2)
        assert_eq(res, sol)