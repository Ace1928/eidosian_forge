from __future__ import annotations
import contextlib
import copy
import pathlib
import re
import xml.etree.ElementTree
from unittest import mock
import pytest
import math
import operator
import os
import time
import warnings
from functools import reduce
from io import StringIO
from operator import add, sub
from threading import Lock
from tlz import concat, merge
from tlz.curried import identity
import dask
import dask.array as da
from dask.array.chunk import getitem
from dask.array.core import (
from dask.array.numpy_compat import NUMPY_GE_200
from dask.array.reshape import _not_implemented_message
from dask.array.tests.test_dispatch import EncapsulateNDArray
from dask.array.utils import assert_eq, same_keys
from dask.base import compute_as_if_collection, tokenize
from dask.blockwise import broadcast_dimensions
from dask.blockwise import make_blockwise_graph as top
from dask.blockwise import optimize_blockwise
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.layers import Blockwise
from dask.utils import SerializableLock, key_split, parse_bytes, tmpdir, tmpfile
from dask.utils_test import dec, hlg_layer_topological, inc
def test_store_locks():
    _Lock = type(Lock())
    d = da.ones((10, 10), chunks=(2, 2))
    a, b = (d + 1, d + 2)
    at = np.zeros(shape=(10, 10))
    bt = np.zeros(shape=(10, 10))
    lock = Lock()
    v = store([a, b], [at, bt], compute=False, lock=lock)
    assert isinstance(v, Delayed)
    dsk = v.dask
    locks = {vv for v in dsk.values() for vv in v if isinstance(vv, _Lock)}
    assert locks == {lock}
    at = NonthreadSafeStore()
    v = store([a, b], [at, at], lock=lock, scheduler='threads', num_workers=10)
    assert v is None
    at = NonthreadSafeStore()
    assert store(a, at, scheduler='threads', num_workers=10) is None
    assert a.store(at, scheduler='threads', num_workers=10) is None
    at = ThreadSafeStore()
    for i in range(10):
        st = a.store(at, lock=False, scheduler='threads', num_workers=10)
        assert st is None
        if at.max_concurrent_uses > 1:
            break
        if i == 9:
            assert False
    nchunks = sum((math.prod(map(len, e.chunks)) for e in (a, b)))
    for c in (False, True):
        at = np.zeros(shape=(10, 10))
        bt = np.zeros(shape=(10, 10))
        lock = CounterLock()
        v = store([a, b], [at, bt], lock=lock, compute=c, return_stored=True)
        assert all((isinstance(e, Array) for e in v))
        da.compute(v)
        assert lock.acquire_count == lock.release_count
        if c:
            assert lock.acquire_count == 2 * nchunks
        else:
            assert lock.acquire_count == nchunks