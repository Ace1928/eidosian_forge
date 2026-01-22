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
def test_aggregate__single_element_groups(agg_func):
    spec = agg_func
    if spec in ('nunique', 'cov', 'corr'):
        return
    pdf = pd.DataFrame({'a': [1, 1, 3, 3], 'b': [4, 4, 16, 16], 'c': [1, 1, 4, 4], 'd': [1, 1, 3, 3]}, columns=['c', 'b', 'a', 'd'])
    ddf = dd.from_pandas(pdf, npartitions=3)
    expected = pdf.groupby(['a', 'd']).agg(spec)
    if spec in {'mean', 'var'}:
        expected = expected.astype(float)
    shuffle_method = {'shuffle_method': 'tasks', 'split_out': 2} if agg_func == 'median' else {}
    assert_eq(expected, ddf.groupby(['a', 'd']).agg(spec, **shuffle_method))