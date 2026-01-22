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
def test_groupby_shift_with_freq(shuffle_method):
    pdf = pd.DataFrame(dict(a=[1, 2, 3, 4, 5, 6], b=[0, 0, 0, 1, 1, 1]), index=pd.date_range(start='20100101', periods=6))
    ddf = dd.from_pandas(pdf, npartitions=3)
    df_result = pdf.groupby(pdf.index).shift(periods=-2, freq='D')
    assert_eq(df_result, ddf.groupby(ddf.index).shift(periods=-2, freq='D', meta=df_result), check_freq=False)
    df_result = pdf.groupby('b').shift(periods=-2, freq='D')
    assert_eq(df_result, ddf.groupby('b').shift(periods=-2, freq='D', meta=df_result), check_freq=shuffle_method != 'disk')