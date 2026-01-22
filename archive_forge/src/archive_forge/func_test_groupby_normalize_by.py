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
def test_groupby_normalize_by():
    full = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [4, 5, 6, 3, 2, 1, 0, 0, 0]}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
    d = dd.from_pandas(full, npartitions=3)
    if DASK_EXPR_ENABLED:
        assert d.groupby('a').by == ['a']
        assert d.groupby(d['a']).by == ['a']
    else:
        assert d.groupby('a').by == 'a'
        assert d.groupby(d['a']).by == 'a'
        assert d.groupby(d['a'] > 2).by._name == (d['a'] > 2)._name
    assert d.groupby(['a', 'b']).by == ['a', 'b']
    assert d.groupby([d['a'], d['b']]).by == ['a', 'b']
    assert d.groupby([d['a'], 'b']).by == ['a', 'b']