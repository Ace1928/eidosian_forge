from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.fixture
def ddf_right_double(df_right):
    return dd.from_pandas(df_right, npartitions=2, sort=False)