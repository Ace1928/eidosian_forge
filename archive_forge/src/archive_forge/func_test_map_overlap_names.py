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
@pytest.mark.parametrize('enforce_metadata', [True, False])
@pytest.mark.parametrize('transform_divisions', [True, False])
@pytest.mark.parametrize('align_dataframes', [True, False])
def test_map_overlap_names(npartitions, enforce_metadata, transform_divisions, align_dataframes):
    ddf = dd.from_pandas(df, npartitions)
    res = ddf.map_overlap(shifted_sum, 0, 3, 0, 3, c=2, align_dataframes=align_dataframes, transform_divisions=transform_divisions, enforce_metadata=enforce_metadata)
    res2 = ddf.map_overlap(shifted_sum, 0, 3, 0, 3, c=2, align_dataframes=align_dataframes, transform_divisions=transform_divisions, enforce_metadata=enforce_metadata)
    assert set(res.dask) == set(res2.dask)
    res3 = ddf.map_overlap(shifted_sum, 0, 3, 0, 3, c=3, align_dataframes=align_dataframes, transform_divisions=transform_divisions, enforce_metadata=enforce_metadata)
    assert res3._name != res._name
    diff = res3.dask.keys() - res.dask.keys()
    assert len(diff) == npartitions
    res4 = ddf.map_overlap(shifted_sum, 3, 0, 0, 3, c=2)
    assert res4._name != res._name