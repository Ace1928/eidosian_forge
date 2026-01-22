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
@pytest.mark.parametrize('key', ['a', ('a-123', h1)])
def test_persist_delayed_custom_key(key):
    d = Delayed(key, {key: 'b', 'b': 1})
    assert d.compute() == 1
    dp = d.persist()
    assert dp.compute() == 1
    assert dp.key == key
    assert dict(dp.dask) == {key: 1}