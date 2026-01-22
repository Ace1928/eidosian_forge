from __future__ import annotations
import datetime
import functools
import operator
import pickle
from array import array
import pytest
from tlz import curry
from dask import get
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable
from dask.utils import (
from dask.utils_test import inc
def test_ensure_unicode():
    data = [b'1', '1', memoryview(b'1'), bytearray(b'1'), array('B', b'1')]
    for d in data:
        result = ensure_unicode(d)
        assert isinstance(result, str)
        assert result == '1'