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
def test_has_keyword():

    def foo(a, b, c=None):
        pass
    assert has_keyword(foo, 'a')
    assert has_keyword(foo, 'b')
    assert has_keyword(foo, 'c')
    bar = functools.partial(foo, a=1)
    assert has_keyword(bar, 'b')
    assert has_keyword(bar, 'c')