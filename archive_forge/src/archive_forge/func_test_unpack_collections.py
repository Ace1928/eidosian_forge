from __future__ import annotations
import dataclasses
import inspect
import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import Executor
from operator import add, mul
from typing import NamedTuple
import pytest
from tlz import merge, partial
import dask
import dask.bag as db
from dask.base import (
from dask.delayed import Delayed, delayed
from dask.diagnostics import Profiler
from dask.highlevelgraph import HighLevelGraph
from dask.utils import tmpdir, tmpfile
from dask.utils_test import dec, import_or_none, inc
def test_unpack_collections():

    @dataclasses.dataclass
    class ADataClass:
        a: int

    class ANamedTuple(NamedTuple):
        a: int
    a = delayed(1) + 5
    b = a + 1
    c = a + 2

    def build(a, b, c, iterator):
        t = (a, b, {'a': a, a: b, 'b': [1, 2, [b]], 'c': 10, 'd': (c, 2), 'e': {a, 2, 3}, 'f': OrderedDict([('a', a)]), 'g': ADataClass(a=a), 'h': (ADataClass, a), 'i': ANamedTuple(a=a)}, iterator)
        return t
    args = build(a, b, c, (i for i in [a, b, c]))
    collections, repack = unpack_collections(*args)
    assert len(collections) == 3
    result = repack(['~a', '~b', '~c'])
    sol = build('~a', '~b', '~c', ['~a', '~b', '~c'])
    assert result == sol
    collections, repack = unpack_collections(*args, traverse=False)
    assert len(collections) == 2
    assert repack(collections) == args
    collections, repack = unpack_collections(1, 2, {'a': 3})
    assert not collections
    assert repack(collections) == (1, 2, {'a': 3})

    def fail(*args):
        raise ValueError("Shouldn't have been called")
    collections, repack = unpack_collections(a, (fail, 1), [(fail, 2, 3)], traverse=False)
    repack(collections)
    repack([(fail, 1)])