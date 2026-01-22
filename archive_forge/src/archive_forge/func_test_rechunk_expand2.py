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
def test_rechunk_expand2():
    a, b = (3, 2)
    orig = np.random.default_rng().uniform(0, 1, a ** b).reshape((a,) * b)
    for off, off2 in product(range(1, a - 1), range(1, a - 1)):
        old = ((a - off, off),) * b
        x = da.from_array(orig, chunks=old)
        new = ((a - off2, off2),) * b
        assert np.all(x.rechunk(chunks=new).compute() == orig)
        if a - off - off2 > 0:
            new = ((off, a - off2 - off, off2),) * b
            y = x.rechunk(chunks=new).compute()
            assert np.all(y == orig)