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
@pytest.mark.parametrize('split_every', [1, 8])
@pytest.mark.parametrize('split_out', [2, 32])
def test_shuffle_aggregate(shuffle_method, split_out, split_every):
    pdf = pd.DataFrame({'a': [1, 2, 3, 1, 1, 2, 4, 3, 7] * 100, 'b': [4, 2, 7, 3, 3, 1, 1, 1, 2] * 100, 'c': [0, 1, 2, 3, 4, 5, 6, 7, 8] * 100, 'd': [3, 2, 1, 3, 2, 1, 2, 6, 4] * 100}, columns=['c', 'b', 'a', 'd'])
    ddf = dd.from_pandas(pdf, npartitions=100)
    spec = {'b': 'mean', 'c': ['min', 'max']}
    result = ddf.groupby(['a', 'b'], sort=False).agg(spec, split_out=split_out, split_every=split_every, shuffle_method=shuffle_method)
    expect = pdf.groupby(['a', 'b']).agg(spec)
    expect['b', 'mean'] = expect['b', 'mean'].astype(result['b', 'mean'].dtype)
    assert_eq(expect, result)