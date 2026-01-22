from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
def test_map_overlap_errors():
    with pytest.raises(ValueError):
        ddf.map_overlap(shifted_sum, 0.5, 3, 0, 2, c=2)
    with pytest.raises(ValueError):
        ddf.map_overlap(shifted_sum, 0, -5, 0, 2, c=2)
    with pytest.raises(NotImplementedError):
        ddf.map_overlap(shifted_sum, 0, 100, 0, 100, c=2).compute()
    with pytest.raises(TypeError):
        ddf.map_overlap(shifted_sum, pd.Timedelta('1s'), pd.Timedelta('1s'), 0, 2, c=2)
    with pytest.raises(TypeError):
        ddf.map_overlap(shifted_sum, '1s', '1s', 0, 2, c=2)