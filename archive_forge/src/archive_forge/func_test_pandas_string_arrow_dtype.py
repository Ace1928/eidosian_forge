from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
@pytest.mark.parametrize('cls_name', ['Series', 'DataFrame', 'Index'])
def test_pandas_string_arrow_dtype(cls_name):
    pytest.importorskip('pyarrow')
    cls = getattr(pd, cls_name)
    s = cls(['x' * 100000, 'y' * 50000], dtype='string[pyarrow]')
    assert 150000 < sizeof(s) < 155000