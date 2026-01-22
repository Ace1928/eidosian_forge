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
def test_replace_name_in_keys():
    assert replace_name_in_key('foo', {}) == 'foo'
    assert replace_name_in_key('foo', {'bar': 'baz'}) == 'foo'
    assert replace_name_in_key('foo', {'foo': 'bar', 'baz': 'asd'}) == 'bar'
    assert replace_name_in_key('foo-123', {'foo-123': 'bar-456'}) == 'bar-456'
    assert replace_name_in_key(('foo-123', h1, h2), {'foo-123': 'bar'}) == ('bar', h1, h2)
    with pytest.raises(TypeError):
        replace_name_in_key(1, {})
    with pytest.raises(TypeError):
        replace_name_in_key((), {})
    with pytest.raises(TypeError):
        replace_name_in_key((1,), {})