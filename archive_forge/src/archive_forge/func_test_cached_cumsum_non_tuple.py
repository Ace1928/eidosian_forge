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
def test_cached_cumsum_non_tuple():
    a = [1, 2, 3]
    assert cached_cumsum(a) == (1, 3, 6)
    a[1] = 4
    assert cached_cumsum(a) == (1, 5, 8)