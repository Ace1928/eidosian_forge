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
def test_ndeepmap():
    L = 1
    assert ndeepmap(0, inc, L) == 2
    L = [1]
    assert ndeepmap(0, inc, L) == 2
    L = [1, 2, 3]
    assert ndeepmap(1, inc, L) == [2, 3, 4]
    L = [[1, 2], [3, 4]]
    assert ndeepmap(2, inc, L) == [[2, 3], [4, 5]]
    L = [[[1, 2], [3, 4, 5]], [[6], []]]
    assert ndeepmap(3, inc, L) == [[[2, 3], [4, 5, 6]], [[7], []]]