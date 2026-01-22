from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_consistent_across_sizes(generator_class):
    x1 = generator_class(123).random(20, chunks=20)
    x2 = generator_class(123).random(100, chunks=20)[:20]
    x3 = generator_class(123).random(200, chunks=20)[:20]
    assert_eq(x1, x2)
    assert_eq(x1, x3)