from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
@pytest.mark.parametrize('shape,chunks', [[(4,), (2,)], [(4, 4), (2, 2)], [(4, 4), (4, 2)]])
def test_dont_concatenate_single_chunks(shape, chunks):
    x = da.ones(shape, chunks=shape)
    y = x.rechunk(chunks)
    dsk = dict(y.dask)
    assert not any((funcname(task[0]).startswith('concat') for task in dsk.values() if dask.istask(task)))