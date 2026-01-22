from __future__ import annotations
import contextlib
import decimal
import warnings
import weakref
import xml.etree.ElementTree
from datetime import datetime, timedelta
from itertools import product
from operator import add
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from pandas.errors import PerformanceWarning
from pandas.io.formats import format as pandas_format
import dask
import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby
from dask import delayed
from dask.base import compute_as_if_collection
from dask.blockwise import fuse_roots
from dask.dataframe import _compat, methods
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import (
from dask.dataframe.utils import (
from dask.datasets import timeseries
from dask.utils import M, is_dataframe_like, is_series_like, put_lines
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.parametrize('npartitions', [1, 4, 20])
@pytest.mark.parametrize('split_every', [2, 5])
@pytest.mark.parametrize('split_out', [None, 1, 5, 20])
def test_hash_split_unique(npartitions, split_every, split_out):
    if DASK_EXPR_ENABLED and split_out is None:
        pytest.skip('no longer supported')
    from string import ascii_lowercase
    s = pd.Series(np.random.choice(list(ascii_lowercase), 1000, replace=True))
    ds = dd.from_pandas(s, npartitions=npartitions)
    dropped = ds.unique(split_every=split_every, split_out=split_out)
    dsk = dropped.__dask_optimize__(dropped.dask, dropped.__dask_keys__())
    from dask.core import get_deps
    dependencies, dependents = get_deps(dsk)
    if not DASK_EXPR_ENABLED:
        assert len([k for k, v in dependencies.items() if not v]) == npartitions
    assert dropped.npartitions == (split_out or 1)
    assert sorted(dropped.compute(scheduler='sync')) == sorted(s.unique())