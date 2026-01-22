from __future__ import annotations
import pickle
from datetime import timedelta
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import (
from xarray.tests import (
from xarray.tests.test_coding_times import (
@requires_cftime
def test_indexing_in_dataframe_loc(df, index, scalar_args, range_args):
    expected = pd.Series([1], name=index[0])
    for arg in scalar_args:
        result = df.loc[arg]
        assert result.equals(expected)
    expected = pd.DataFrame([1, 2], index=index[:2])
    for arg in range_args:
        result = df.loc[arg]
        assert result.equals(expected)