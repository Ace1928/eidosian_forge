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
def test_cached_cumsum():
    a = (1, 2, 3, 4)
    x = cached_cumsum(a)
    y = cached_cumsum(a, initial_zero=True)
    assert x == (1, 3, 6, 10)
    assert y == (0, 1, 3, 6, 10)