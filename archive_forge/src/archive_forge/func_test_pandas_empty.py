from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
@requires_pandas
def test_pandas_empty():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': ['a' * 100, 'b' * 100, 'c' * 100]}, index=[10, 20, 30])
    empty = df.head(0)
    assert sizeof(empty) > 0
    assert sizeof(empty.x) > 0
    assert sizeof(empty.y) > 0
    assert sizeof(empty.index) > 0