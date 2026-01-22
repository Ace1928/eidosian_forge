from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
@requires_pandas
@pytest.mark.parametrize('cls_name', ['Series', 'DataFrame', 'Index'])
@pytest.mark.parametrize('dtype', [object, 'string[python]'])
def test_pandas_object_dtype(dtype, cls_name):
    cls = getattr(pd, cls_name)
    s1 = cls([f'x{i:3d}' for i in range(1000)], dtype=dtype)
    assert sizeof('x000') * 1000 < sizeof(s1) < 2 * sizeof('x000') * 1000
    x = 'x' * 100000
    y = 'y' * 100000
    z = 'z' * 100000
    w = 'w' * 100000
    s2 = cls([x, y, z, w] * 1000, dtype=dtype)
    assert 400000 < sizeof(s2) < 500000
    s3 = cls([x, y, z, w], dtype=dtype)
    s4 = cls([x, y, z, x], dtype=dtype)
    s5 = cls([x, x, x, x], dtype=dtype)
    assert sizeof(s5) < sizeof(s4) < sizeof(s3)