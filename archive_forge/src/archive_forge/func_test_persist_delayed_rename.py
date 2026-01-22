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
@pytest.mark.parametrize('key,rename,new_key', [('a', {}, 'a'), ('a', {'c': 'd'}, 'a'), ('a', {'a': 'b'}, 'b'), (('a-123', h1), {'a-123': 'b-123'}, ('b-123', h1))])
def test_persist_delayed_rename(key, rename, new_key):
    d = Delayed(key, {key: 1})
    assert d.compute() == 1
    rebuild, args = d.__dask_postpersist__()
    dp = rebuild({new_key: 2}, *args, rename=rename)
    assert dp.compute() == 2
    assert dp.key == new_key
    assert dict(dp.dask) == {new_key: 2}