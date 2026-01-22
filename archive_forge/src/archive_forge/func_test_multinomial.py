from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_multinomial(generator_class):
    for size, chunks in [(5, 3), ((5, 4), (2, 3))]:
        x = generator_class().multinomial(20, [1 / 6.0] * 6, size=size, chunks=chunks)
        y = np.random.multinomial(20, [1 / 6.0] * 6, size=size)
        assert x.shape == y.shape == x.compute().shape