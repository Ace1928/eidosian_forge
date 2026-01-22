from __future__ import annotations
import contextlib
import operator
import warnings
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import _concat
from dask.dataframe.utils import (
def test_categorical_set_index(shuffle_method):
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': ['a', 'b', 'b', 'c']})
    df['y'] = pd.Categorical(df['y'], categories=['a', 'b', 'c'], ordered=True)
    a = dd.from_pandas(df, npartitions=2)
    with dask.config.set(scheduler='sync'):
        b = a.set_index('y', npartitions=a.npartitions)
        d1, d2 = (b.get_partition(0), b.get_partition(1))
        assert list(d1.index.compute()) == ['a']
        assert list(sorted(d2.index.compute())) == ['b', 'b', 'c']
        b = a.set_index(a.y, npartitions=a.npartitions)
        d1, d2 = (b.get_partition(0), b.get_partition(1))
        assert list(d1.index.compute()) == ['a']
        assert list(sorted(d2.index.compute())) == ['b', 'b', 'c']
        b = a.set_index('y', divisions=['a', 'b', 'c'], npartitions=a.npartitions)
        d1, d2 = (b.get_partition(0), b.get_partition(1))
        assert list(d1.index.compute()) == ['a']
        assert list(sorted(d2.index.compute())) == ['b', 'b', 'c']