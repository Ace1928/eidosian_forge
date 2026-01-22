from __future__ import annotations
import itertools
import pickle
from functools import partial
import pytest
import dask
from dask.base import tokenize
from dask.core import get_dependencies
from dask.local import get_sync
from dask.optimization import (
from dask.utils import apply, partial_by_order
from dask.utils_test import add, inc
def test_fuse_stressed():

    def f(*args):
        return args
    d = {'array-original-27b9f9d257a80fa6adae06a98faf71eb': 1, ('cholesky-upper-26a6b670a8aabb7e2f8936db7ccb6a88', 0, 0): (f, ('cholesky-26a6b670a8aabb7e2f8936db7ccb6a88', 0, 0)), ('cholesky-26a6b670a8aabb7e2f8936db7ccb6a88', 1, 0): (f, ('cholesky-upper-26a6b670a8aabb7e2f8936db7ccb6a88', 0, 1)), ('array-27b9f9d257a80fa6adae06a98faf71eb', 0, 0): (f, 'array-original-27b9f9d257a80fa6adae06a98faf71eb', (slice(0, 10, None), slice(0, 10, None))), ('cholesky-upper-26a6b670a8aabb7e2f8936db7ccb6a88', 1, 0): ('cholesky-26a6b670a8aabb7e2f8936db7ccb6a88', 0, 1), ('cholesky-26a6b670a8aabb7e2f8936db7ccb6a88', 1, 1): (f, (f, ('array-27b9f9d257a80fa6adae06a98faf71eb', 1, 1), (f, [('cholesky-lt-dot-26a6b670a8aabb7e2f8936db7ccb6a88', 1, 0, 1, 0)]))), ('cholesky-lt-dot-26a6b670a8aabb7e2f8936db7ccb6a88', 1, 0, 1, 0): (f, ('cholesky-26a6b670a8aabb7e2f8936db7ccb6a88', 1, 0), ('cholesky-upper-26a6b670a8aabb7e2f8936db7ccb6a88', 0, 1)), ('array-27b9f9d257a80fa6adae06a98faf71eb', 0, 1): (f, 'array-original-27b9f9d257a80fa6adae06a98faf71eb', (slice(0, 10, None), slice(10, 20, None))), ('cholesky-upper-26a6b670a8aabb7e2f8936db7ccb6a88', 1, 1): (f, ('cholesky-26a6b670a8aabb7e2f8936db7ccb6a88', 1, 1)), ('cholesky-26a6b670a8aabb7e2f8936db7ccb6a88', 0, 1): (f, (10, 10)), ('array-27b9f9d257a80fa6adae06a98faf71eb', 1, 1): (f, 'array-original-27b9f9d257a80fa6adae06a98faf71eb', (slice(10, 20, None), slice(10, 20, None))), ('cholesky-upper-26a6b670a8aabb7e2f8936db7ccb6a88', 0, 1): (f, ('cholesky-26a6b670a8aabb7e2f8936db7ccb6a88', 0, 0), ('array-27b9f9d257a80fa6adae06a98faf71eb', 0, 1)), ('cholesky-26a6b670a8aabb7e2f8936db7ccb6a88', 0, 0): (f, ('array-27b9f9d257a80fa6adae06a98faf71eb', 0, 0))}
    keys = {('cholesky-upper-26a6b670a8aabb7e2f8936db7ccb6a88', 0, 0), ('cholesky-upper-26a6b670a8aabb7e2f8936db7ccb6a88', 0, 1), ('cholesky-upper-26a6b670a8aabb7e2f8936db7ccb6a88', 1, 0), ('cholesky-upper-26a6b670a8aabb7e2f8936db7ccb6a88', 1, 1)}
    rv = fuse(d, keys=keys, ave_width=2, rename_keys=True)
    assert rv == with_deps(rv[0])