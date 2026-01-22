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
def test_iter_chunks():
    sizes = [14, 8, 5, 9, 7, 9, 1, 19, 8, 19]
    assert list(iter_chunks(sizes, 19)) == [[14], [8, 5], [9, 7], [9, 1], [19], [8], [19]]
    assert list(iter_chunks(sizes, 28)) == [[14, 8, 5], [9, 7, 9, 1], [19, 8], [19]]
    assert list(iter_chunks(sizes, 67)) == [[14, 8, 5, 9, 7, 9, 1], [19, 8, 19]]