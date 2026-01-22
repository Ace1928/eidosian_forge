from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
def test_df_groupby_idxmax():
    pdf = pd.DataFrame({'idx': list(range(4)), 'group': [1, 1, 2, 2], 'value': [10, 20, 20, 10]}).set_index('idx')
    ddf = dd.from_pandas(pdf, npartitions=3)
    expected = pd.DataFrame({'group': [1, 2], 'value': [1, 2]}).set_index('group')
    result_pd = pdf.groupby('group').idxmax()
    result_dd = ddf.groupby('group').idxmax()
    assert_eq(result_pd, result_dd)
    assert_eq(expected, result_dd)